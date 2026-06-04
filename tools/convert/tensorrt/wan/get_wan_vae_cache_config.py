# Import inner classes directly to bypass weight loading
import torch
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE_
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import WanVAE_ as WanVAE_2_2

def get_cache_config(vae_cls, model_name, resolutions):
    print(f"--- Analyzing {model_name} ---")
    try:
        # Instantiate VAE on CPU directly
        # Configs from original classes:
        # Wan2.1: dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2
        # Wan2.2: dim=160, dec_dim=256, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2
        
        if "2.1" in model_name:
             inner_model = WanVAE_(dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2)
        else:
             inner_model = WanVAE_2_2(dim=160, dec_dim=256, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2)
        
        inner_model = inner_model.to("cpu")
        
        # Initialize internal cache counters
        inner_model.clear_cache()
        
        encoder = inner_model.encoder
        decoder = inner_model.decoder

        print(f"Encoder Cache Layers: {inner_model._enc_conv_num}")
        print(f"Decoder Cache Layers: {inner_model._conv_num}")

        for h, w in resolutions:
            print(f"\nResolution: {h}x{w}")
            
            # --- Encoder Analysis ---
            # Input to encoder is standard video frame? 
            # Wan2.1: [B, C, T, H, W], usually T=1 for streaming check
            # BUT WanVAE implementation usually takes latent or pixel?
            # Encoder takes pixel. [B, 3, T, H, W]
            # Let's trace one forward pass with hooks or dummy data to get cache shapes
            
            # We can calculate analytically if we know the downsample factors
            # Encoder Downsamples time?
            # Wan2.1 Encoder3d: dims=[128, 256, 512, 512], temp_down=[True, True, False]
            # Let's just run a dummy pass on CPU
            
            # Encoder Input Shape
            # Wan 2,1: encode(x)
            # Patchify happens inside encode?
            # Wan2.1: encode -> self.encoder(x[:, :, :1, ...])
            # Wan2.2: encode -> patchify(x) -> encoder
            
            is_wan2_2 = "2.2" in model_name
            
            dummy_t = 1
            if is_wan2_2:
                # Patchify happens before encoder in Wan2.2? 
                # Wan2.2 encode method does patchify first
                # x = patchify(x, patch_size=2)
                # So encoder input is actually already patchified?
                # Let's look at Wan2.2 Encoder call in encode():
                # self.encoder(x, ...) 
                # So we should feed patchified input to encoder if we are testing encoder directly,
                # OR input to wrapper will handle patchify?
                # The Plan says "Single-step implementation".
                # If we wrap `encoder` then we need to handle inputs as `encoder` expects.
                pass

            # Actually, let's just use the `WanVAE.encoder` (inner module) and see what it expects
            # We will use dummy inputs compatible with what `WanVAE.encode` passes to `encoder`
            
            if is_wan2_2:
                 # Wan 2.2 input is patchified
                 # patchify(x, 2) turns [B, C, T, H, W] -> [B, C*4, T, H/2, W/2]
                 enc_in_c = 12 # 3 * 2 * 2
                 enc_in_h = h // 2
                 enc_in_w = w // 2
            else:
                 enc_in_c = 3
                 enc_in_h = h
                 enc_in_w = w
            
            dummy_enc_in = torch.randn(1, enc_in_c, dummy_t, enc_in_h, enc_in_w)
            
            # Initialize Cache manually
            inner_model.clear_cache()
            # _enc_feat_map is initialized with Nones. We need to see what shapes they become.
            # Running one step
            _ = encoder(dummy_enc_in, feat_cache=inner_model._enc_feat_map, feat_idx=[0])
            
            print("  Encoder Cache Shapes:")
            for i, c in enumerate(inner_model._enc_feat_map):
                if isinstance(c, torch.Tensor):
                    print(f"    idx {i}: {list(c.shape)}")
                else:
                    print(f"    idx {i}: {c}")

            # --- Decoder Analysis ---
            # Decoder takes Latent.
            # Wan2.1: z_dim * 2 ? No, decoder takes z_dim=16?
            # Wan2.1 vae.py: Decoder3d(dim=128, z_dim=4...)
            # Wan2.2 vae_2_2.py: Decoder3d(dec_dim=256, z_dim=16...)
            
            dec_in_c = inner_model.z_dim
            # Decoder Input H/W? 
            # Latent H/W = Input H/W / Stride
            # Check Stride.
            # Wan2.1 Stride? VaE Config usually stride 16 or 32?
            # Wan 2.1: num_res_blocks=2, dim_mult=[1, 2, 4, 4], temporal_down=[True,True,False]
            # Spatial down: Resample in Encoder
            # Encoder3D:
            #  i=0: dim 128->256. downsample3d (temp+spatial) -> scale / 2
            #  i=1: dim 256->512. downsample3d (temp+spatial) -> scale / 4
            #  i=2: dim 512->512. downsample2d (spatial) -> scale / 8
            # Total spatial stride = 8? 
            # Configs often say stride 16.
            
            # Let's trust the Latent Shape calculation in Runner
            # 480p example: 832x480.
            # WanRunner: latent_h = H // stride[1] ??
            # Usually stride is (T, H, W) = (4, 8, 8) for Wan2.1?
            
            stride = 8 # Assumption based on code reading (3 downsamples)
            if is_wan2_2:
                 # Wan 2.2 Decoder is different?
                 # Wan 2.2 Decoder3d output is 12 channels (patchified image?)
                 # unpatchify usage suggests yes.
                 pass

            lat_h = h // stride
            lat_w = w // stride

            dummy_dec_in = torch.randn(1, dec_in_c, 1, lat_h, lat_w)
            
            # Initialize Cache
            inner_model.clear_cache()
            
            # Conv2 projection happens BEFORE decoder loop in VAE
            # x = self.conv2(z)
            # Decoder takes x
            conv2_out = inner_model.conv2(dummy_dec_in)
            
            kwargs = {"feat_cache": inner_model._feat_map, "feat_idx": [0]}
            if is_wan2_2:
                kwargs["first_chunk"] = True # Test first chunk logic
            
            _ = decoder(conv2_out, **kwargs)

            print("  Decoder Cache Shapes:")
            for i, c in enumerate(inner_model._feat_map):
                if isinstance(c, torch.Tensor):
                    print(f"    idx {i}: {list(c.shape)}")
                else:
                    print(f"    idx {i}: {c}")
                    
    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    resolutions = [
        (480, 832), # 480p landscape
        (832, 480), # 480p portrait
        (720, 1280),# 720p
        (1024, 1024) # 1k square
    ]
    
    # Check Wan 2.1
    get_cache_config(WanVAE_, "Wan2.1", resolutions)
    print("="*40)
    # Check Wan 2.2
    get_cache_config(WanVAE_2_2, "Wan2.2", resolutions)
