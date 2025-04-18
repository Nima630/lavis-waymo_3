"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

# from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
        "pretrain_qformer": "configs/models/blip2/blip2_pretrain_qformer.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,  # not used
        use_grad_checkpoint=False,  # not used
        vit_precision="fp16",  # not used
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,  # not used
    ):
        super().__init__()

        # === RGB Encoder ===
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze RGB vision encoder")

        # === LiDAR Encoder (identical structure) ===
        self.lidar_encoder, self.ln_lidar = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.lidar_encoder.named_parameters():
                param.requires_grad = False
            self.lidar_encoder = self.lidar_encoder.eval()
            self.lidar_encoder.train = disabled_train
            logging.info("freeze LiDAR encoder")

        # === Shared Q-Former for RGB and LiDAR ===
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer_lidar, _ = self.init_Qformer(
            num_query_token, self.lidar_encoder.num_features, cross_attention_freq
        )

        # === Projection layers for contrastive loss ===
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.lidar_proj = nn.Linear(self.Qformer_lidar.config.hidden_size, embed_dim)

        # === Matching head (concatenated RGB + LiDAR → binary match) ===
        self.matching_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # === Learnable temperature for contrastive loss ===
        self.temp = nn.Parameter(0.07 * torch.ones([]))


    def forward(self, samples):
        image = samples["image"]
        lidar = samples["lidar"]
        labels = samples["label"]  # 1 = match, 0 = mismatch

        bs = image.size(0)

        # === Encode RGB ===
        rgb_embeds = self.ln_vision(self.visual_encoder(image))
        rgb_atts = torch.ones(rgb_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(bs, -1, -1)

        rgb_query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=rgb_embeds,
            encoder_attention_mask=rgb_atts,
            return_dict=True,
        )
        rgb_feats = F.normalize(self.vision_proj(rgb_query_output.last_hidden_state), dim=-1)

        # === Encode LiDAR ===
        lidar_embeds = self.ln_lidar(self.lidar_encoder(lidar))
        lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(lidar.device)
        lidar_query_output = self.Qformer_lidar.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=lidar_embeds,
            encoder_attention_mask=lidar_atts,
            return_dict=True,
        )
        lidar_feats = F.normalize(self.lidar_proj(lidar_query_output.last_hidden_state), dim=-1)

        # === Contrastive Loss ===
        rgb_feats_all = concat_all_gather(rgb_feats)
        lidar_feats_all = concat_all_gather(lidar_feats)

        sim_r2l = torch.matmul(rgb_feats.unsqueeze(1), lidar_feats_all.unsqueeze(-1)).squeeze().max(-1)[0]
        sim_l2r = torch.matmul(lidar_feats.unsqueeze(1), rgb_feats_all.unsqueeze(-1)).squeeze().max(-1)[0]

        sim_r2l = sim_r2l / self.temp
        sim_l2r = sim_l2r / self.temp

        rank = dist.get_rank()
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(image.device)

        loss_contrastive = (
            F.cross_entropy(sim_r2l, targets, label_smoothing=0.1) +
            F.cross_entropy(sim_l2r, targets, label_smoothing=0.1)
        ) / 2

        # === Matching Head (Binary Classifier) ===
        fused_feats = torch.cat([rgb_feats.mean(dim=1), lidar_feats.mean(dim=1)], dim=-1)
        logits_match = self.matching_head(fused_feats).squeeze()  # Output shape: [batch]
        loss_match = F.binary_cross_entropy_with_logits(logits_match, labels)

        # === Return both losses ===
        total_loss = loss_contrastive + loss_match  # You can weight them if needed

        return BlipOutput(
            loss=total_loss,
            loss_itc=loss_contrastive,
            loss_itm=loss_match,
            loss_lm=torch.tensor(0.0).to(image.device),
        )



    @classmethod
    def from_config(cls, cfg): # ----------------------------------------------------------------------------
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)


    def __init___(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len


    def forward_(self, samples):
        print("[TRACE] forward: start")
        image = samples["image"]
        lidar = samples["lidar"]
        # text = samples["text_input"]

        # 1. Vision encoder
        # print("[TRACE] forward: encoding image via visual_encoder")
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # 2. Q-Former cross-attention with image
        # print("[TRACE] forward: QFormer cross-attention on image features")
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        # 3. Project image to embedding space
        # print("[TRACE] forward: projecting image features")
        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )


        # 1. Vision encoder
        # print("[TRACE] forward: encoding image via visual_encoder")
        lidar_embeds = self.ln_vision(self.visual_encoder(lidar))
        lidar_atts = torch.ones(lidar_embeds.size()[:-1], dtype=torch.long).to(
            lidar.device
        )

        query_tokens_lidar = self.query_tokens.expand(lidar_embeds.shape[0], -1, -1)
        # 2. Q-Former cross-attention with image
        # print("[TRACE] forward: QFormer cross-attention on image features")
        query_output_lidar = self.Qformer.bert(
            query_embeds=query_tokens_lidar,
            encoder_hidden_states=lidar_embeds,
            encoder_attention_mask=lidar_atts,
            use_cache=True,
            return_dict=True,
        )
        # 3. Project image to embedding space
        # print("[TRACE] forward: projecting image features")
        lidar_feats = F.normalize(
            self.vision_proj(query_output_lidar.last_hidden_state), dim=-1
        )
















        # # 4. Tokenize and encode text
        # # print("[TRACE] forward: tokenizing and encoding text")
        # text_tokens = self.tokenizer(
        #     text,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(image.device)


        # 5. Q-Former self-attention (text)
        # print("[TRACE] forward: QFormer self-attention on text features")
        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     attention_mask=text_tokens.attention_mask,
        #     return_dict=True,
        # )

        # # 6. Project text features
        # # print("[TRACE] forward: projecting text features")
        # text_feat = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )

        ###============== Image-text Contrastive ===================###
        # 6. Contrastive loss (ITC)
        # print("[TRACE] forward: computing contrastive loss (ITC)")

        image_feats_all = concat_all_gather(
            image_feats
        ) 
        

        lidar_feats_all = concat_all_gather(
            lidar_feats
        ) 
        
         # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = image.size(0)
        # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
        #     image.device
        # )

        if "image_id" in samples.keys(): #coco retrieval finetuning
            # image_ids = samples["image_id"].view(-1,1)

            image_ids = samples["image_id"]

            # Convert to tensor of integers *only if* your IDs are numeric
            # In your case, they are strings like "coco_522418", so you have two choices:

            # 🔁 Option 1: Use string-to-index mapping (recommended for retrieval tasks)
            # For now, just hash them into dummy integers for testing:
            image_ids = [hash(i) % 10**6 for i in image_ids]  # simple hashing
            image_ids = torch.tensor(image_ids).view(-1, 1).to(samples["image"].device)


            image_ids_all = concat_all_gather(image_ids)
            # pos_idx = torch.eq(image_ids, image_ids_all.t()).float()       
            # sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
            # sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)
            
            # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
            # loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()     
            # loss_itc = (loss_t2i+loss_i2t)/2  
        # else:                     
        #     loss_itc = (
        #         F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #         + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        #     ) / 2

        ###============== Image-text Matching ===================###
        print("[TRACE][ITM] Gathering global text and image embeddings...")
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)

        print("[TRACE][ITM] Masking similarity matrix (diagonal or based on image_id)...")

        with torch.no_grad():
            if "image_id" in samples.keys():
                mask = torch.eq(image_ids, image_ids_all.t())
                sim_t2i.masked_fill_(mask, -10000)
                sim_i2t.masked_fill_(mask, -10000)
            else:    
                sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
                
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )
        print("[TRACE] ITM: QFormer forward pass for image-text matching")
        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )
        print("[TRACE] ITM: itm_head projection (ITM logits)")
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        # decoder_input_ids = text_tokens.input_ids.clone()
        # decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        # labels = decoder_input_ids.masked_fill(
        #     decoder_input_ids == self.tokenizer.pad_token_id, -100
        # )

        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        # lm_output = self.Qformer(
        #     decoder_input_ids,
        #     attention_mask=attention_mask,
        #     past_key_values=query_output.past_key_values,
        #     return_dict=True,
        #     labels=labels,
        # )

        # loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_itm, # + loss_itc +  loss_lm,
            loss_itc= 0, #loss_itc,
            loss_itm=loss_itm,
            loss_lm= 0, #loss_lm,
        )
