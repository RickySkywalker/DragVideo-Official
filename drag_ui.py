import os
import gradio as gr

from DragVideo.utils.ui_utils import get_points_start, get_points_end, undo_points
from DragVideo.utils.ui_utils import clear_all, train_lora_interface, run_drag
from DragVideo.utils.ui_utils import store_video, propagate
from DragVideo.utils.ui_utils import sam_refine, clear_click

from DragVideo.args import ui_args

LENGTH=ui_args.window_size # length of the square area displaying/editing images
BACKUP_DREAMBOOTH_MODELS = ui_args.dreambooth_model_list

with gr.Blocks() as demo:
    # layout definition
    with gr.Row():
        gr.Markdown("""
        # Official Implementation of [DragVideo](https://dragvideo.github.io/)
        """)

    masks = gr.State(value=None) # store all masks
    logits = gr.State(value=None) # store all logits
    handle_points = gr.State(value=None) # store all points
    target_points = gr.State(value=None) # store all points

    # UI components for editing real videos
    with gr.Tab(label="Step 1: Pre-Editing"):
        original_image_start = gr.State(value=None) # store original input image
        original_image_end = gr.State(value=None) # store original input image
        original_video = gr.State(value=None) # store original input video
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">input video</p>""")
                video = gr.Video(label="Input video",interactive = True,
                    show_label=True).style(height=LENGTH, width=LENGTH)
                train_lora_button = gr.Button("Train LoRA")
                
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">key frames preview</p>""")
                preview_kps_video = gr.Video(label="Key frame preview",
                    show_label=True).style(height=LENGTH, width=LENGTH)
                clear_all_button = gr.Button("Clear All")
                
        
        # general parameters
        with gr.Row():
            # data_path = gr.Textbox(value="./data_tmp", label="data path") #use to store video key frames to train lora
            lora_batch = gr.Number(value=ui_args.drag_batch, label="LoRA batch", info="number of frames to train LoRA")
            inverse_batch = gr.Number(value=ui_args.inv_batch, label="inverse batch", info="number of frames to inverse")
            drag_batch = gr.Number(value=ui_args.drag_batch, label="edit length", info="number of frames to edit")
            max_len = gr.Number(value=ui_args.max_len, label="max length", info="max length of video")
            fps = gr.Number(value=24, label="fps", info="frame per second (auto)")
            kps = gr.Number(value=ui_args.kps, label="kps", info="key frame per second")

        with gr.Row():
            prompt = gr.Textbox(value=ui_args.prompt, label="Prompt")
            local_models_dir = 'AnimateDiff/models/DreamBooth_LoRA'
            local_models_choice = \
                [os.path.join(local_models_dir,d) for d in os.listdir(local_models_dir) if os.path.isfile(os.path.join(local_models_dir,d)) and not d.endswith('.txt')]
            model_path = gr.Dropdown(value="realisticVisionV60B1_v51VAE.safetensors", #need to modified to ours
                label="Personalized Model Path",
                choices=BACKUP_DREAMBOOTH_MODELS + local_models_choice,
            )
            lora_path = gr.Textbox(value=ui_args.lora_path, label="LoRA path")
            save_dir = gr.Textbox(value=ui_args.save_path, label="save dir")

        with gr.Row():
            lora_step = gr.Number(value=100, label="LoRA training steps", precision=0)
            lora_lr = gr.Number(value=0.0005, label="LoRA learning rate")
            lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)
            lora_status_bar = gr.Textbox(label="display LoRA training status")
    with gr.Tab(label="Step 2: Editing Video"):
        click_state = gr.State([[],[]])
        selected_mask = gr.State(value=None) # store mask
        selected_logit = gr.State(value=None) # store logit
        selected_points_start = gr.State([]) # store points
        selected_points_end = gr.State([]) # store points
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points on start frame</p>""")
                point_start = gr.Image(type="numpy", label="Click Points",
                    show_label=True).style(height=LENGTH, width=LENGTH) # for points clicking
                undo_button_start = gr.Button("Undo point") 

                gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points on end frame</p>""")
                point_end = gr.Image(type="numpy", label="Click Points",
                    show_label=True).style(height=LENGTH, width=LENGTH) # for points clicking
                undo_button_end = gr.Button("Undo point")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Define the mask</p>""")
                template_frame = gr.Image(type="pil", interactive=True, elem_id="template_frame").style(height=LENGTH, width=LENGTH)
                point_prompt = gr.Radio(
                    choices=["Positive",  "Negative"],
                    value="Positive",
                    label="Point prompt",
                    interactive=True)
                clear_button_click = gr.Button(value="Clear clicks", interactive=True)
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">preview propagation</p>""")
                preview_video = gr.Video(label="Preview",
                    show_label=True).style(height=LENGTH, width=LENGTH)
                propagate_button = gr.Button("Propagate Point & Mask")
                run_button = gr.Button("Run Drag")
                gr.Markdown("""<p style="text-align: center; font-size: 20px">drag result</p>""")
                output_video = gr.Video(label="Output",
                    show_label=True).style(height=LENGTH, width=LENGTH)
                

        # algorithm specific parameters
        with gr.Row():
            n_pix_step = gr.Number(
                value=ui_args.drag_steps,
                label="number of pixel steps",
                info="Number of gradient descent (motion supervision) steps on latent.",
                precision=0)
            lam = gr.Number(value=ui_args.lam, label="lam", info="regularization strength on unmasked areas")
            # n_actual_inference_step = gr.Number(value=40, label="optimize latent step", precision=0)
            inversion_strength = gr.Slider(0, 1.0,
                value=0.8,
                label="inversion strength",
                info="The latent at [inversion-strength * total-sampling-steps] is optimized for dragging.")
            latent_lr = gr.Number(value=ui_args.lr, label="latent lr")
            start_step = gr.Number(value=0, label="start_step", precision=0, visible=False)
            start_layer = gr.Number(value=10, label="start_layer", precision=0, visible=False)
            # TODO this is used for ablation
            guidance_scale = gr.Number(value=ui_args.guidance_scale, label="guidance scale", info="The scale of the guidance loss.")
            radius = gr.Number(value=10, label="radius", info="The extend radius of the mask.")
            
            

    # event definition
    # event for dragging user-input real image
    video.change(
        store_video,
        [video, max_len, drag_batch, kps, save_dir],
        [original_video, preview_kps_video, preview_video, output_video, original_image_start, original_image_end, point_start, point_end, template_frame, fps]
    )
    point_start.select(
        get_points_start,
        [original_image_start, selected_points_start, click_state, template_frame, selected_mask, selected_logit],
        [point_start, template_frame, selected_mask, selected_logit],
    )
    point_end.select(
        get_points_end,
        [original_image_end, selected_points_end],
        [point_end],
    )
    undo_button_start.click(
        undo_points,
        [original_image_start],
        [point_start, selected_points_start]
    )
    undo_button_end.click(
        undo_points,
        [original_image_end],
        [point_end, selected_points_end]
    )

    template_frame.select(
        sam_refine,
        inputs=[original_image_start, point_prompt, click_state],
        outputs=[template_frame, selected_mask, selected_logit]
    )

    clear_button_click.click(
        clear_click,
        inputs = [original_image_start, click_state,],
        outputs = [template_frame,click_state],
    )

    train_lora_button.click(
        train_lora_interface,
        [original_video,
        prompt,
        model_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_rank,
        lora_batch],
        [lora_status_bar]
    )
    propagate_button.click(
        propagate,
        [original_video,
        selected_mask,
        prompt,
        selected_points_start,
        selected_points_end,
        kps,
        save_dir,
        radius,
        ],
        [preview_video, handle_points, target_points, masks, logits]
    )   
    run_button.click(
        run_drag,
        [original_video,
        masks,
        prompt,
        handle_points,
        target_points,
        kps,
        drag_batch,
        inverse_batch,
        inversion_strength,
        lam,
        latent_lr,
        n_pix_step,
        model_path,
        lora_path,
        guidance_scale,
        start_step,
        start_layer,
        save_dir
        ],
        [output_video]
    )
    clear_all_button.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [
        point_start,
        point_end,
        template_frame,
        preview_video,
        preview_kps_video,
        output_video,
        video,
        selected_points_start,
        selected_points_end,
        selected_mask]
    )




demo.queue().launch(share=True)
