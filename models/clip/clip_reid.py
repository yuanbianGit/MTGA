import torch
import torch.nn as nn
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from timm.models.layers import trunc_normal_


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # print(x.shape)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # 这一行代码选择序列中的某个特定位置的向量，通常是句子中的某个特定标记，比如 [CLS] 标记，用来代表整个句子的意义。
        # text.argmax(dim=-1) 找到这个（结束）特殊标记的位置，然后提取对应的向量。接着，将该向量与 self.text_projection 进行矩阵乘法，以得到最终的文本向量表示。
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class build_clip(nn.Module):
    def __init__(self, num_classes):
        super(build_clip, self).__init__()
        self.model_name = 'ViT-B-16'
        self.cos_layer = False
        self.neck = 'bnneck'
        self.neck_feat = 'before'
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes

        self.sie_coe = 1.0

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((256 - 16) // 16 + 1)
        self.w_resolution = int((128 - 16) // 16 + 1)
        self.vision_stride_size = 16

        # TODO Load CLIP
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        dataset_name = 'dukemtmc'
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x=None, is_training=False):

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        # else:
        #     if self.neck_feat == 'after':
        #         # print("Test with feature after BN")
        #         cls_score = self.classifier(feat)
        #         return cls_score, torch.cat([feat, feat_proj], dim=1)
        #     else:
        #         cls_score_proj = self.classifier_proj(feat_proj)
        #         return cls_score_proj,torch.cat([img_feature, img_feature_proj], dim=1)
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                cls_score = self.classifier(feat)
                return [cls_score,feat]
            else:
                cls_score = self.classifier(img_feature)
                return [cls_score,img_feature]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


# NOTE This is what i Need
class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


class build_transformer_att(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer_att, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        # Load CLIP
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        # 这里修改prompt拼接
        self.prompt_learner = Prompt_Cat(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        self.token_embbdings = clip_model.token_embedding
        self.prompt_text_gender = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                          output_dim=clip_model.transformer.width,
                                          n_layer=3)

        self.prompt_text_upper = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                         output_dim=clip_model.transformer.width,
                                         n_layer=3)
        self.prompt_text_under = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                         output_dim=clip_model.transformer.width,
                                         n_layer=3)
        self.prompt_text_hair = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                        output_dim=clip_model.transformer.width,
                                        n_layer=3)
        self.prompt_text_shoes = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                         output_dim=clip_model.transformer.width,
                                         n_layer=3)
        self.prompt_text_bag = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                       output_dim=clip_model.transformer.width,
                                       n_layer=3)

    def forward(self, x=None, label=None):
        image_features_last, image_features, image_features_proj = self.image_encoder(x)
        # FIXME 要注意，这里的feature都是应用的哪个，proj的是clip里用通过一层linear层的，可以认为，att时还是要用前边两个
        # print(image_features_last.size())
        # print(image_features.size())
        # print(image_features_proj.size())
        # torch.Size([64, 129, 768])
        # torch.Size([64, 129, 768])
        # torch.Size([64, 129, 512])
        if self.model_name == 'RN50':
            img_f_clip = image_features_proj[0]
        elif self.model_name == 'ViT-B-16':
            img_f_clip = image_features_proj[:, 0]

        prompt_gender = self.prompt_text_gender(img_f_clip).unsqueeze(dim=1)
        prompt_upper = self.prompt_text_upper(img_f_clip).unsqueeze(dim=1)
        prompt_under = self.prompt_text_under(img_f_clip).unsqueeze(dim=1)
        prompt_hair = self.prompt_text_hair(img_f_clip).unsqueeze(dim=1)
        prompt_shoes = self.prompt_text_shoes(img_f_clip).unsqueeze(dim=1)
        prompt_bag = self.prompt_text_bag(img_f_clip).unsqueeze(dim=1)  # [64, 512]
        # FIXME 加入类别信息，做分类引导
        prompts = self.prompt_learner(label, [prompt_gender, prompt_upper, prompt_under, prompt_hair, prompt_shoes,
                                              prompt_bag])
        text_f = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        if self.model_name == 'RN50':
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
        elif self.model_name == 'ViT-B-16':
            img_feature = image_features[:, 0]
        feat = self.bottleneck(img_feature)
        # print(prompt_bag.size())
        #         torch.Size([64, 1, 512])
        prompt_embedding = torch.cat(
            [prompt_gender, prompt_upper, prompt_under, prompt_hair, prompt_shoes, prompt_bag]).squeeze()

        return img_f_clip, text_f, feat, prompts, prompt_embedding

        '''
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)
        '''

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_att_1(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer_att_1, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        # Load CLIP
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        # 这里修改prompt拼接
        self.prompt_learner = Prompt_Cat_1(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        self.token_embbdings = clip_model.token_embedding
        self.prompt_text_semantic = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                            output_dim=clip_model.transformer.width,
                                            n_layer=3)

    def forward(self, x=None, label=None):
        image_features_last, image_features, image_features_proj = self.image_encoder(x)
        # FIXME 要注意，这里的feature都是应用的哪个，proj的是clip里用通过一层linear层的，可以认为，att时还是要用前边两个
        # print(image_features_last.size())
        # print(image_features.size())
        # print(image_features_proj.size())
        # torch.Size([64, 129, 768])
        # torch.Size([64, 129, 768])
        # torch.Size([64, 129, 512])
        if self.model_name == 'RN50':
            img_f_clip = image_features_proj[0]
        elif self.model_name == 'ViT-B-16':
            img_f_clip = image_features_proj[:, 0]

        prompt_semantic = self.prompt_text_semantic(img_f_clip).unsqueeze(dim=1)

        # FIXME 加入类别信息，做分类引导
        prompts = self.prompt_learner(label, [prompt_semantic])
        text_f = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        if self.model_name == 'RN50':
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
        elif self.model_name == 'ViT-B-16':
            img_feature = image_features[:, 0]
        feat = self.bottleneck(img_feature)
        # print(prompt_bag.size())
        #         torch.Size([64, 1, 512])
        prompt_embedding = torch.cat([prompt_semantic]).squeeze()

        return img_f_clip, text_f, feat, prompts, prompt_embedding

        '''
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)
        '''

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_att_2(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer_att_2, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        # Load CLIP
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        # 这里修改prompt拼接
        self.prompt_learner = Prompt_Cat_2(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        self.token_embbdings = clip_model.token_embedding
        self.prompt_text_semantic = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                            output_dim=clip_model.transformer.width,
                                            n_layer=3)

    def forward(self, x=None, label=None):
        image_features_last, image_features, image_features_proj = self.image_encoder(x)

        if self.model_name == 'RN50':
            img_f_clip = image_features_proj[0]
        elif self.model_name == 'ViT-B-16':
            img_f_clip = image_features_proj[:, 0]

        prompt_semantic = self.prompt_text_semantic(img_f_clip).unsqueeze(dim=1)

        # FIXME 加入类别信息，做分类引导
        prompts = self.prompt_learner(label, [prompt_semantic])
        text_f = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        if self.model_name == 'RN50':
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
        elif self.model_name == 'ViT-B-16':
            img_feature = image_features[:, 0]
        feat = self.bottleneck(img_feature)
        # print(prompt_bag.size())
        #         torch.Size([64, 1, 512])
        prompt_embedding = torch.cat([prompt_semantic]).squeeze()

        return img_f_clip, text_f, feat, prompts, prompt_embedding

        '''
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)
        '''

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_att_3(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer_att_3, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        # Load CLIP
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        # 这里修改prompt拼接
        self.prompt_learner = Prompt_Cat_3(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        self.token_embbdings = clip_model.token_embedding

        self.prompt_text_style = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                         output_dim=clip_model.transformer.width,
                                         n_layer=3)
        self.prompt_text_gender = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                          output_dim=clip_model.transformer.width,
                                          n_layer=3)

        self.prompt_text_upper = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                         output_dim=clip_model.transformer.width,
                                         n_layer=3)
        self.prompt_text_under = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                         output_dim=clip_model.transformer.width,
                                         n_layer=3)
        self.prompt_text_hair = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                        output_dim=clip_model.transformer.width,
                                        n_layer=3)
        self.prompt_text_shoes = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                         output_dim=clip_model.transformer.width,
                                         n_layer=3)
        self.prompt_text_bag = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                       output_dim=clip_model.transformer.width,
                                       n_layer=3)

    def forward(self, x=None, label=None):
        image_features_last, image_features, image_features_proj = self.image_encoder(x)

        if self.model_name == 'RN50':
            img_f_clip = image_features_proj[0]
        elif self.model_name == 'ViT-B-16':
            img_f_clip = image_features_proj[:, 0]
        prompt_style = self.prompt_text_style(img_f_clip).unsqueeze(dim=1)
        prompt_gender = self.prompt_text_gender(img_f_clip).unsqueeze(dim=1)
        prompt_upper = self.prompt_text_upper(img_f_clip).unsqueeze(dim=1)
        prompt_under = self.prompt_text_under(img_f_clip).unsqueeze(dim=1)
        prompt_hair = self.prompt_text_hair(img_f_clip).unsqueeze(dim=1)
        prompt_shoes = self.prompt_text_shoes(img_f_clip).unsqueeze(dim=1)
        prompt_bag = self.prompt_text_bag(img_f_clip).unsqueeze(dim=1)  # [64, 512]
        # FIXME 加入类别信息，做分类引导
        prompts = self.prompt_learner(label, [prompt_style, prompt_gender, prompt_upper, prompt_under, prompt_hair,
                                              prompt_shoes, prompt_bag])
        text_f = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        if self.model_name == 'RN50':
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
        elif self.model_name == 'ViT-B-16':
            img_feature = image_features[:, 0]
        feat = self.bottleneck(img_feature)
        # print(prompt_bag.size())
        #         torch.Size([64, 1, 512])
        prompt_embedding = torch.cat(
            [prompt_style, prompt_gender, prompt_upper, prompt_under, prompt_hair, prompt_shoes, prompt_bag]).squeeze()

        return img_f_clip, text_f, feat, prompts, prompt_embedding

        '''
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)
        '''

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_att_4(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer_att_4, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        # Load CLIP
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        # 这里修改prompt拼接
        self.prompt_learner = Prompt_Cat_2(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        self.token_embbdings = clip_model.token_embedding
        self.prompt_text_semantic = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                            output_dim=clip_model.transformer.width,
                                            n_layer=3)
        self.prompt_text_style = IM2TEXT(embed_dim=clip_model.visual.output_dim, middle_dim=512,
                                         output_dim=clip_model.transformer.width,
                                         n_layer=3)

    def forward(self, x=None, label=None):
        image_features_last, image_features, image_features_proj = self.image_encoder(x)

        if self.model_name == 'RN50':
            img_f_clip = image_features_proj[0]
        elif self.model_name == 'ViT-B-16':
            img_f_clip = image_features_proj[:, 0]

        prompt_style = self.prompt_text_style(img_f_clip).unsqueeze(dim=1)
        prompt_semantic = self.prompt_text_semantic(img_f_clip).unsqueeze(dim=1)

        # FIXME 加入类别信息，做分类引导
        prompts = self.prompt_learner(label, [prompt_style, prompt_semantic])
        text_f = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        if self.model_name == 'RN50':
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
        elif self.model_name == 'ViT-B-16':
            img_feature = image_features[:, 0]
        feat = self.bottleneck(img_feature)
        # print(prompt_bag.size())
        #         torch.Size([64, 1, 512])
        prompt_embedding = torch.cat([prompt_style, prompt_semantic]).squeeze()

        return img_f_clip, text_f, feat, prompts, prompt_embedding

        '''
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)
        '''

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))





from models.clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 4, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class Prompt_Cat(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            # FIXME 考虑怎么把这个X 换为 (X X)
            # ctx_init = "A X wearing X top, X out"
            ctx_init = "A photo of a X wearing X on top, " \
                       "X underneath, X hairstyle, X shoes, and carrying X."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 1
        # tokenize出来的是加了开始和截止符号，空格不算，标点符号算，343是X
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # NOTE 这里纯为了加载，没用
        self.register_buffer("token_prefix", embedding[:, :4 + 1, :])
        self.register_buffer("token_suffix", embedding[:, 4 + 1 + 4:, :])

        self.register_buffer("token_prefix_1", embedding[:, :5, :])
        self.register_buffer("token_prefix_2", embedding[:, 6, :].unsqueeze(dim=1))
        self.register_buffer("token_prefix_3", embedding[:, 8:11, :])
        self.register_buffer("token_prefix_4", embedding[:, 12:14, :])
        self.register_buffer("token_prefix_5", embedding[:, 15:17, :])
        self.register_buffer("token_prefix_6", embedding[:, 18:22, :])
        self.register_buffer("token_suffix_my", embedding[:, 23:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, prom_list):
        # FIXME 要把cls_ctx建立起来！
        # cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        # (B,text_len,512)
        prefix_1 = self.token_prefix_1.expand(b, -1, -1)
        prefix_2 = self.token_prefix_2.expand(b, -1, -1)
        prefix_3 = self.token_prefix_3.expand(b, -1, -1)
        prefix_4 = self.token_prefix_4.expand(b, -1, -1)
        prefix_5 = self.token_prefix_5.expand(b, -1, -1)
        prefix_6 = self.token_prefix_6.expand(b, -1, -1)

        suffix = self.token_suffix_my.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix_1,  # (n_cls, 1, dim)
                prom_list[0],
                prefix_2,  # (n_cls, 1, dim)
                prom_list[1],
                prefix_3,  # (n_cls, 1, dim)
                prom_list[2],
                prefix_4,  # (n_cls, 1, dim)
                prom_list[3],
                prefix_5,  # (n_cls, 1, dim)
                prom_list[4],
                prefix_6,  # (n_cls, 1, dim)
                prom_list[5],
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class Prompt_Cat_1(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            # ctx_init = "A photo of a X wearing X on top, " \
            #                        "X underneath, X hairstyle, X shoes, and carrying X."
            ctx_init = "A photo of X."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 1
        # tokenize出来的是加了开始和截止符号，空格不算，标点符号算，343是X
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # NOTE 这里纯为了加载，没用
        self.register_buffer("token_prefix", embedding[:, :n_cls_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_cls_ctx + 1 + n_cls_ctx:, :])

        self.register_buffer("token_prefix_1", embedding[:, :4, :])
        self.register_buffer("token_prefix_2", embedding[:, 5:, :])

        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, prom_list):
        # FIXME 要把cls_ctx建立起来！
        # cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        # (B,text_len,512)
        prefix_1 = self.token_prefix_1.expand(b, -1, -1)
        prefix_2 = self.token_prefix_2.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix_1,  # (n_cls, 1, dim)
                prom_list[0],
                prefix_2,  # (n_cls, 1, dim)
            ],
            dim=1,
        )

        return prompts


class Prompt_Cat_2(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            # ctx_init = "A photo of a X wearing X on top, " \
            #                        "X underneath, X hairstyle, X shoes, and carrying X."
            ctx_init = "A photo of a X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 1
        # tokenize出来的是加了开始和截止符号，空格不算，标点符号算，343是X
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # NOTE 这里纯为了加载，没用
        self.register_buffer("token_prefix", embedding[:, :n_cls_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_cls_ctx + 1 + n_cls_ctx:, :])

        self.register_buffer("token_prefix_1", embedding[:, :5, :])
        self.register_buffer("token_prefix_2", embedding[:, 6:, :])

        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, prom_list):
        # FIXME 要把cls_ctx建立起来！
        # cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        # (B,text_len,512)
        prefix_1 = self.token_prefix_1.expand(b, -1, -1)
        prefix_2 = self.token_prefix_2.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix_1,  # (n_cls, 1, dim)
                prom_list[0],
                prefix_2,  # (n_cls, 1, dim)
            ],
            dim=1,
        )

        return prompts


class Prompt_Cat_3(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            # FIXME 考虑怎么把这个X 换为 (X X)
            # ctx_init = "A X wearing X top, X out"
            ctx_init = "A X style photo of a X wearing X on top, " \
                       "X underneath, X hairstyle, X shoes, and carrying X."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 1
        # tokenize出来的是加了开始和截止符号，空格不算，标点符号算，343是X
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # NOTE 这里纯为了加载，没用
        self.register_buffer("token_prefix", embedding[:, :4 + 1, :])
        self.register_buffer("token_suffix", embedding[:, 4 + 1 + 4:, :])

        self.register_buffer("token_prefix_1", embedding[:, :2, :])
        self.register_buffer("token_prefix_2", embedding[:, 3:7, :])
        self.register_buffer("token_prefix_3", embedding[:, 8, :].unsqueeze(dim=1))
        self.register_buffer("token_prefix_4", embedding[:, 10:13, :])
        self.register_buffer("token_prefix_5", embedding[:, 14:16, :])
        self.register_buffer("token_prefix_6", embedding[:, 17:19, :])
        self.register_buffer("token_prefix_7", embedding[:, 20:24, :])
        self.register_buffer("token_suffix_8", embedding[:, 25:, :])

        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, prom_list):
        # FIXME 要把cls_ctx建立起来！
        # cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        # (B,text_len,512)

        prefix_1 = self.token_prefix_1.expand(b, -1, -1)
        prefix_2 = self.token_prefix_2.expand(b, -1, -1)
        prefix_3 = self.token_prefix_3.expand(b, -1, -1)
        prefix_4 = self.token_prefix_4.expand(b, -1, -1)
        prefix_5 = self.token_prefix_5.expand(b, -1, -1)
        prefix_6 = self.token_prefix_6.expand(b, -1, -1)
        prefix_7 = self.token_prefix_7.expand(b, -1, -1)
        suffix = self.token_suffix_8.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix_1,  # (n_cls, 1, dim)
                prom_list[0],
                prefix_2,  # (n_cls, 1, dim)
                prom_list[1],
                prefix_3,  # (n_cls, 1, dim)
                prom_list[2],
                prefix_4,  # (n_cls, 1, dim)
                prom_list[3],
                prefix_5,  # (n_cls, 1, dim)
                prom_list[4],
                prefix_6,  # (n_cls, 1, dim)
                prom_list[5],
                prefix_7,  # (n_cls, 1, dim)
                prom_list[6],
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class Prompt_Cat_4(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            # FIXME 考虑怎么把这个X 换为 (X X)
            # ctx_init = "A X wearing X top, X out"
            ctx_init = "A X style photo of a X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 1
        # tokenize出来的是加了开始和截止符号，空格不算，标点符号算，343是X
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # NOTE 这里纯为了加载，没用
        self.register_buffer("token_prefix", embedding[:, :4 + 1, :])
        self.register_buffer("token_suffix", embedding[:, 4 + 1 + 4:, :])

        self.register_buffer("token_prefix_1", embedding[:, :2, :])
        self.register_buffer("token_prefix_2", embedding[:, 3:7, :].unsqueeze(dim=1))
        self.register_buffer("token_prefix_3", embedding[:, 8:, :])

        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, prom_list):
        # FIXME 要把cls_ctx建立起来！
        # cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        # (B,text_len,512)

        prefix_1 = self.token_prefix_1.expand(b, -1, -1)
        prefix_2 = self.token_prefix_2.expand(b, -1, -1)
        prefix_3 = self.token_prefix_3.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix_1,  # (n_cls, 1, dim)
                prom_list[0],
                prefix_2,  # (n_cls, 1, dim)
                prom_list[1],
                prefix_3,  # (n_cls, 1, dim)

            ],
            dim=1,
        )

        return prompts