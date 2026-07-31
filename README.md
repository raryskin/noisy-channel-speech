# Eye-gaze dynamics reveal the adaptive nature of real-time noisy-channel inference in speech comprehension
This repository contains code, data, and full R Markdown manuscript for the manuscript titled "Eye-gaze dynamics reveal the adaptive nature of real-time noisy-channel inference in speech comprehension"

## Reproducing the manuscript
1. Clone the repository and open in RStudio.
2. Download all files from the [OSF repository](https://osf.io/ybxru/overview?view_only=33cb61764a7c4d7baa8fdbb5b28cd1f3) and put them in the `data` folder. 
3. Verify that the following libraries are installed: `papaja`, `tidyverse`, `cmdstanr`, `bayesplot`,`tidybayes`,`loo`, `patchwork`, `magick`, `cowplot`, `ggrepel`.
4. Set all chunks in `manuscript_sections/stan_model_fitting.Rmd` to `eval = TRUE`.
5. Knit `noisy_speech_manuscript.Rmd`.

## Contact
For questions regarding the code, data, or manuscript, please contact Rachel Ryskin at rryskin@gmail.com
