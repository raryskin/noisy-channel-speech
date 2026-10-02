# Eye-gaze dynamics reveal the adaptive nature of real-time noisy-channel inference in speech comprehension
This repository contains all the code to reproduce the manuscript titled "Eye-gaze dynamics reveal the adaptive nature of real-time noisy-channel inference in speech comprehension"

## Reproducing the manuscript
1. Clone the repository.
2. Download all data files from the [OSF repository](https://osf.io/ybxru/overview?view_only=33cb61764a7c4d7baa8fdbb5b28cd1f3) and put them in a folder called `data` in the main directory. 
3. Verify that the following libraries are installed: `papaja`, `tidyverse`, `cmdstanr`, `bayesplot`,`tidybayes`,`loo`, `patchwork`, `magick`, `cowplot`, `ggrepel`.
4. Inside `manuscript_sections/`, create a folder called `rdata/` and set all chunks in `manuscript_sections/stan_model_fitting.Rmd` to `eval = TRUE`.
5. Knit `noisy_speech_manuscript.Rmd`. (This will write some `.rds` files to the `rdata` folder. If you want to knit the manuscript again, you can set the chunks back to `eval = FALSE` in order to save time.)

## Contact
For questions regarding the code, data, or manuscript, please contact Rachel Ryskin at rryskin@gmail.com
