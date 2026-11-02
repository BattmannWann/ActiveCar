## Refresh Rate & Reaction Time

- Retinal-to-cortical transduction takes roughly 30 to 60ms
- Saccadic targeting and fixation stabilisation takes around 150 to 200 ms (this would be good to get papers to gauge on)

- Assetto Corsa (AC) vehicle physics operate at 25-50Hz (a 40-20ms step interval; set in the config through `ego_sampling_freq` and in AC through the refresh rate in the settings).

- **How to make this realistic?**: Keep the environment physics and motor policy stepping at 25-50Hz, while updating the visual gaze fixation and foveal crop at a realistic biological interval of 100ms (10Hz) 

---

## Architecture

> *NOTE: To obtain a more in-depth explanation, please consult the jupyter notebook.*

<br>

- **Visual Processing Pipeline**: Three modules which comprise an end-to-end active vision system. These are the:

    - `SaliencyEngine`: In other words, the *Global Perception*. This module has been abstracted to allow for varying classes to be implemented in a plug-and-play fashion with the framework. Simply create a new class, and change the engine that the framework references as its saliency engine. 

        - The current base engine is that of the *Spectral Residual Model*, which dictates that the information within the image can be split into the *Prior Knowledge / Background* and the *Innovation / Saliency / Anomaly*

        - It uses the fourier spectral residual analysis to find where the most interesting visual stimulus is located within the image 

            ```txt
                    [ Input Image ]
                            │
                            ▼
                [ 2D Fourier Transform ] ───► Extract Amplitude & Phase
                            │
                            ├───► Log Amplitude ──► Smooth (Local Average)
                            │           │                    │
                            │           └─────────┬──────────┘
                            │                     ▼
                            │            Subtract Average
                            │                     │
                            │                     ▼
                            │            [ Spectral Residual ]
                            │                     │
                            └───► Combine with original Phase
                                            │
                                            ▼
                            [ Inverse 2D Fourier Transform ]
                                            │
                                            ▼
                            [ Gaussian Blur & Scale [0, 1] ]
                                            │
                                            ▼
                                    [ Saliency Heatmap ]

            ```

    <br>

    - `FovealExtractor`: The *Local Attention*. This module crops a high-resolution sub-region around the most salient part (a coordinate returned by the SaliencyEngine), converting the pixel location into normalised gaze coordinates. 

        - This is based on the ***FOVEATED*** human visual system: A broad, low-resolution *peripheral* (often processed as low-res saliency maps), and a high-resolution, sub-region target (called the *fovea*), used for fine-grained feature analysis. 

        - The FovealExtractor takes the target focal point (a coordinate $(cx, cy)$), in this case the peak salient coordinate produced by the previous SaliencyEngine, crops a fixed-size square parch around it (clipping to prevent out-of-bounds), and converts the pixel coordinates into a normalised gaze vector:

            ```txt
                [ Full Image (Width x Height) ]
            ┌──────────────────────────────┐
            │                              │
            │        (cx, cy) Target       │
            │           ┌───────┐          │
            │           │ Crop  │          │  ───►  [ High-Res Foveal Crop ] (96 x 96)
            │           └───────┘          │
            │                              │
            └──────────────────────────────┘
                            │
                            ▼
            [ Coordinate Normalization ]   ───►  Gaze Vector [norm_x, norm_y] in [-1.0, 1.0]

            ```

    <br>

    - `FovealVisionEncoder`: The *Semantic Feature Extraction* module. This translates the raw, high-resolution cropped foveal image and gaze coordinates into a 514-dimensional representation suitable for downstream machine learning (ML) tasks (such as as RL policies, visual navigation, or active target tracking). 

        - The 514-d vector is a result of the pretrained ResNet-18 deep learning backbone, which is used to extract deep semantic visual features (512 features + gaze coords = 514). 

            ```txt
            [ Foveal Crop (96x96x3) ]           [ Gaze Coordinates [norm_x, norm_y] ]
                        │                                          │
                        ▼                                          │
            [ ImageNet Normalization ]                             │
                        │                                          │
                        ▼                                          │
            [ Frozen ResNet-18 Backbone ]                          │
            (Conv Layers ──► Layer 4 ──► Adaptive AvgPool)         │
                        │                                          │
                        ▼                                          │
            [ 512-d Visual Features ]                              │
                        │                                          │
                        └───────────────────┬──────────────────────┘
                                            ▼
                                [ Feature Concatenation ]
                                            │
                                            ▼
                            [ Output State Vector (514-d) ]
                            [ 512 Visual + 2 Positional ]

            ```

<br>

- **Gymnasium Integration**: Normally, RL algorithms require massive neural networks, compute, and memory to process high-resolution raw pixel matrices directly. Instead, this framework bypasses the overhead by acting as a ***foveated perception filter***:

    - It intercepts the raw visual observations from the environment, runs them through the (active vision) visual processing pipeline, and converts the high-dimensional video streams into a compact 514-dimensional continuous vector for RL agent consumption

    
        ```txt
                [ Gymnasium Environment / Screen Grabber ]
                                    │
                                    ▼
                                [ Raw RGB Frame ]
                                    │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        [ Saliency Engine (Frequency) ]        [ Foveal Extractor ]
                    │                                     │
                    ▼                                     ▼
        [ Peak Salient Point (x, y) ]  ───►  [ High-Res Patch Crop ]
                                                        │
                                                        ▼
                                            [ ResNet-18 Vision Encoder ]
                                                        │
                                                        ▼
                                            [ 514-d State Vector ]
                                                        │
                                                        ▼
                                            [ RL Agent Policy (SAC) ]

        ```

    - For comparison, instead of the $6.22 x 10^{6}$ parameters an agent would have to process from a standard 1080p RGB image (i.e. $1920 \times 1080 \times 3$), it only has to process a 514 continuous valued vector (as $12,000 \times$ compression factor)