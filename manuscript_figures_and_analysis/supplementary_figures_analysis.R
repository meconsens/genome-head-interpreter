library(qs)
library(tidyverse)
library(Biostrings)
library(ggpubr)
library(reticulate)
library(yardstick)
pd <- import("pandas")


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Options ----
options(dplyr.summarise.inform = FALSE)
dnabert_layer_head_factors <- c(expand.grid(paste0('head', 0:11), paste0('layer', 0:11))) %>% 
  as_tibble() %>% 
  dplyr::mutate(layer = paste(Var2, Var1, sep='-')) %>% 
  dplyr::pull(layer)
dnabert_layer_factors <- str_split_i(dnabert_layer_head_factors, "-", 1) %>% unique()
dnabert_head_factors <- str_split_i(dnabert_layer_head_factors, "-", 2) %>% unique()
nt_layer_head_factors <- c(expand.grid(paste0('head', 0:15), paste0('layer', 0:28))) %>% 
  as_tibble() %>% 
  dplyr::mutate(layer = paste(Var2, Var1, sep='-')) %>% 
  dplyr::pull(layer)
nt_layer_factors <- str_split_i(nt_layer_head_factors, "-", 1) %>% unique()
nt_head_factors <- str_split_i(nt_layer_head_factors, "-", 2) %>% unique()


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# By layer-head-tag metrics ----
open_layer_head_tag_scores <- function(x){
  
  files <- c("dnabert_tata" = "files/intermediate/custom_tata_dnabert_ft_layer_head_tag_metrics.qs",
             "dnabert_fake" = "files/intermediate/fake_tata_dnabert_ft_layer_head_tag_metrics.qs",
             "dnabert_enhancer" = "files/intermediate/enhancer_dnabert_ft_layer_head_tag_metrics.qs",
             
             "nt_tata" = "files/intermediate/custom_tata_nt_ft_layer_head_tag_metrics.qs",
             "nt_fake" = "files/intermediate/fake_tata_nt_ft_layer_head_tag_metrics.qs",
             "nt_enhancer" = "files/intermediate/enhancer_nt_ft_layer_head_tag_metrics.qs")
  
  layer_head_tag_scores <- set_names(names(files)) %>% 
    map(\(x) {qread(files[[x]])}) %>% 
    list_rbind(names_to = "model_name") %>% 
    dplyr::mutate(model = str_split_i(model_name, "_", 1),
                  dataset = str_split_i(model_name, "_", 2)) %>% 
    dplyr::select(model_name, model, dataset, id, negative, positive, distance_diag) %>% 
    dplyr::group_by(model_name) %>% 
    dplyr::mutate(sig = ifelse(distance_diag > (mean(distance_diag)+sd(distance_diag*3)), "Significantly changed", "No change")) %>% 
    dplyr::ungroup()
}
layer_head_tag_scores_df <- open_layer_head_tag_scores()

## Supp Figure S1 ----
layer_head_tag_scores_plotlist <- layer_head_tag_scores_df %>% 
  dplyr::mutate(id = str_replace_all(id, "_", "-"),
                model_name = factor(model_name,
                                    levels = c("dnabert_tata", "dnabert_fake", "dnabert_enhancer",
                                               "nt_tata", "nt_fake", "nt_enhancer"),
                                    labels = c("DNABERT - TATA", "DNABERT - Fake TATA", "DNABERT - Enhancers",
                                               "Nucleotide Transformer - TATA", "Nucleotide Transformer - Fake TATA", "Nucleotide Transformer - Enhancers"))) %>% 
  dplyr::group_by(model_name) %>% 
  group_map(\(tb.f, key) {
    tb.f %>% 
      ggplot(aes(x = positive, y = negative)) +
      ggrastr::geom_point_rast(aes(color = sig, alpha = sig)) +
      geom_abline() +
      geom_hline(aes(yintercept = 0), linetype = "dashed") +
      geom_vline(aes(xintercept = 0), linetype = "dashed") +
      ggrepel::geom_label_repel(data = . %>% 
                                  dplyr::filter(sig == "Significantly changed") %>%
                                  dplyr::slice_max(order_by = distance_diag, n = 3),
                                aes(label = id),
                                box.padding = 0.8,
                                alpha = 0.9,
                                size = 3,
                                force = 2,
                                seed = 4321) +
      xlim(c(-1,1)) +
      ylim(c(-1,1)) +
      scale_color_manual(values = c("Significantly changed" = "#bc4749", "No change" = "black")) +
      scale_alpha_manual(values = c("Significantly changed" = 1, "No change" = 0.7)) +
      labs(title = key$model_name,
           x = "Positive controls",
           y = "Negative controls") +
      theme_minimal() +
      theme(plot.title = element_text(size = 10, hjust = 0.5),
            panel.border = element_rect(fill = NA),
            strip.text = element_text(size = 9),
            axis.title = element_text(size = 9),
            axis.text = element_text(size = 8),
            legend.title = element_blank(),
            legend.position = "top")
  })

gg_supS1 <- ggpubr::ggarrange(layer_head_tag_scores_plotlist[[1]], layer_head_tag_scores_plotlist[[2]],
                              layer_head_tag_scores_plotlist[[3]], NULL,
                              layer_head_tag_scores_plotlist[[4]], layer_head_tag_scores_plotlist[[5]],
                              layer_head_tag_scores_plotlist[[6]], NULL,
                              labels = c("A", "", "", "", "B", "", "", ""),
                              ncol = 2, nrow = 4,
                              common.legend = TRUE,
                              legend = "top")
gg_supS1 <- ggpubr::annotate_figure(gg_supS1,
                                    top = "Difference between k-mer and head mean attention scores\n(k-mer_score - head_score)")
ggsave('SuppFigureS1_A4.pdf',
       plot = gg_supS1,
       width = 8.3,  # in inches
       height = 11.7, # in inches
       units = "in",
       dpi = 300,
       bg = "white")


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Z-score metrics ----
open_zscores <- function(){
  zscore_files <- c("dnabert_ft_tata_0" = "files/z_score_matrices/DNABERT_TATA/finetuned/label_specific/0_centered_z_scores.csv",
                    "dnabert_ft_tata_1" = "files/z_score_matrices/DNABERT_TATA/finetuned/label_specific/1_centered_z_scores.csv",
                    "dnabert_pt_tata_0" = "files/z_score_matrices/DNABERT_TATA/pretrained/label_specific/0_centered_z_scores.csv",
                    "dnabert_pt_tata_1" = "files/z_score_matrices/DNABERT_TATA/pretrained/label_specific/1_centered_z_scores.csv",
                    "dnabert_ri_tata_0" = "files/z_score_matrices/DNABERT_TATA/random_init/label_specific/0_centered_z_scores.csv",
                    "dnabert_ri_tata_1" = "files/z_score_matrices/DNABERT_TATA/random_init/label_specific/1_centered_z_scores.csv",
                    "dnabert_rpt_tata_0" = "files/z_score_matrices/DNABERT_TATA/random/label_specific/0_centered_z_scores.csv",
                    "dnabert_rpt_tata_1" = "files/z_score_matrices/DNABERT_TATA/random/label_specific/1_centered_z_scores.csv",
                    
                    "dnabert_ft_fake_0" = "files/z_score_matrices/DNABERT_fake_TATA/finetuned/label_specific/0_centered_z_scores.csv",
                    "dnabert_ft_fake_1" = "files/z_score_matrices/DNABERT_fake_TATA/finetuned/label_specific/1_centered_z_scores.csv",
                    "dnabert_pt_fake_0" = "files/z_score_matrices/DNABERT_fake_TATA/pretrained/label_specific/0_centered_z_scores.csv",
                    "dnabert_pt_fake_1" = "files/z_score_matrices/DNABERT_fake_TATA/pretrained/label_specific/1_centered_z_scores.csv",
                    "dnabert_ri_fake_0" = "files/z_score_matrices/DNABERT_fake_TATA/random_init/label_specific/0_centered_z_scores.csv",
                    "dnabert_ri_fake_1" = "files/z_score_matrices/DNABERT_fake_TATA/random_init/label_specific/1_centered_z_scores.csv",
                    "dnabert_rpt_fake_0" = "files/z_score_matrices/DNABERT_fake_TATA/random/label_specific/0_centered_z_scores.csv",
                    "dnabert_rpt_fake_1" = "files/z_score_matrices/DNABERT_fake_TATA/random/label_specific/1_centered_z_scores.csv",
                    
                    "dnabert_ft_enhancer_0" = "files/z_score_matrices/DNABERT_enhancers/finetuned/label_specific/0_centered_z_scores.csv",
                    "dnabert_ft_enhancer_1" = "files/z_score_matrices/DNABERT_enhancers/finetuned/label_specific/1_centered_z_scores.csv",
                    "dnabert_pt_enhancer_0" = "files/z_score_matrices/DNABERT_enhancers/pretrained/label_specific/0_centered_z_scores.csv",
                    "dnabert_pt_enhancer_1" = "files/z_score_matrices/DNABERT_enhancers/pretrained/label_specific/1_centered_z_scores.csv",
                    "dnabert_ri_enhancer_0" = "files/z_score_matrices/DNABERT_enhancers/random_init/label_specific/0_centered_z_scores.csv",
                    "dnabert_ri_enhancer_1" = "files/z_score_matrices/DNABERT_enhancers/random_init/label_specific/1_centered_z_scores.csv",
                    "dnabert_rpt_enhancer_0" = "files/z_score_matrices/DNABERT_enhancers/random/label_specific/0_centered_z_scores.csv",
                    "dnabert_rpt_enhancer_1" = "files/z_score_matrices/DNABERT_enhancers/random/label_specific/1_centered_z_scores.csv",
                    
                    "nt_ft_tata_0" = "files/z_score_matrices/NT_TATA/finetuned/label_specific/0_centered_z_scores.csv",
                    "nt_ft_tata_1" = "files/z_score_matrices/NT_TATA/finetuned/label_specific/1_centered_z_scores.csv",
                    "nt_pt_tata_0" = "files/z_score_matrices/NT_TATA/pretrained/label_specific/0_centered_z_scores.csv",
                    "nt_pt_tata_1" = "files/z_score_matrices/NT_TATA/pretrained/label_specific/1_centered_z_scores.csv",
                    "nt_ri_tata_0" = "files/z_score_matrices/NT_TATA/random_init/label_specific/0_centered_z_scores.csv",
                    "nt_ri_tata_1" = "files/z_score_matrices/NT_TATA/random_init/label_specific/1_centered_z_scores.csv",
                    
                    "nt_ft_fake_0" = "files/z_score_matrices/NT_fake_TATA/finetuned/label_specific/0_centered_z_scores.csv",
                    "nt_ft_fake_1" = "files/z_score_matrices/NT_fake_TATA/finetuned/label_specific/1_centered_z_scores.csv",
                    "nt_pt_fake_0" = "files/z_score_matrices/NT_fake_TATA/pretrained/label_specific/0_centered_z_scores.csv",
                    "nt_pt_fake_1" = "files/z_score_matrices/NT_fake_TATA/pretrained/label_specific/1_centered_z_scores.csv",
                    "nt_ri_fake_0" = "files/z_score_matrices/NT_fake_TATA/random_init/label_specific/0_centered_z_scores.csv",
                    "nt_ri_fake_1" = "files/z_score_matrices/NT_fake_TATA/random_init/label_specific/1_centered_z_scores.csv",
                    
                    "nt_ft_enhancer_0" = "files/z_score_matrices/NT_enhancers/finetuned/label_specific/0_centered_z_scores.csv",
                    "nt_ft_enhancer_1" = "files/z_score_matrices/NT_enhancers/finetuned/label_specific/1_centered_z_scores.csv",
                    "nt_pt_enhancer_0" = "files/z_score_matrices/NT_enhancers/pretrained/label_specific/0_centered_z_scores.csv",
                    "nt_pt_enhancer_1" = "files/z_score_matrices/NT_enhancers/pretrained/label_specific/1_centered_z_scores.csv",
                    "nt_ri_enhancer_0" = "files/z_score_matrices/NT_enhancers/random_init/label_specific/0_centered_z_scores.csv",
                    "nt_ri_enhancer_1" = "files/z_score_matrices/NT_enhancers/random_init/label_specific/1_centered_z_scores.csv",
                    
                    "scgpt_ft_ms_global" = "files/z_score_matrices/scgpt_ms/finetuned_centered_z_scores.csv",
                    "scgpt_pt_ms_global" = "files/z_score_matrices/scgpt_ms/pretrained_centered_z_scores.csv", 
                    "scgpt_ri_ms_global" = "files/z_score_matrices/scgpt_ms/random_init_centered_z_scores.csv", 
                    
                    "scgpt_ft_pancreas_global" = "files/z_score_matrices/scgpt_pancreas/finetuned_centered_z_scores.csv",
                    "scgpt_pt_pancreas_global" = "files/z_score_matrices/scgpt_pancreas/pretrained_centered_z_scores.csv", 
                    "scgpt_ri_pancreas_global" = "files/z_score_matrices/scgpt_pancreas/random_init_centered_z_scores.csv")
  
  zscore_df <- names(zscore_files) %>% map(\(x) {
    read_csv(zscore_files[[x]], show_col_types = FALSE) %>% 
      dplyr::rename(layer = `...1`) %>% 
      tidyr::pivot_longer(cols = -layer,
                          names_to = "feature",
                          values_to = "zscore") %>% 
      dplyr::mutate(model = str_split_i(x, "_", 1),
                    training = str_split_i(x, "_", 2),
                    dataset = str_split_i(x, "_", 3),
                    label = str_split_i(x, "_", 4))
  }) %>% bind_rows()
}
zscore_df <- open_zscores()

## Supp Figure S2 ----
gg_zscore_pt_ri_df <- zscore_df %>% 
  dplyr::filter(training %in% c("pt", "ri")) %>% 
  tidyr::pivot_wider(names_from = training,
                     values_from = zscore) %>% 
  dplyr::mutate(distance_diag = abs(pt - ri)/sqrt(2),
                label = factor(label,
                                  levels = c("0", "1", "global"),
                                  labels = c("Negative controls", "Positive controls", "Global")),
                head_name = glue::glue("{layer}-{feature}"),
                model_name = glue::glue("{model}_{dataset}"),
                model_name = factor(model_name,
                                    levels = c("dnabert_tata", "dnabert_fake", "dnabert_enhancer",
                                               "nt_tata", "nt_fake", "nt_enhancer",
                                               "scgpt_ms", "scgpt_pancreas"),
                                    labels = c("DNABERT - TATA", "DNABERT - Fake TATA", "DNABERT - Enhancers",
                                               "Nucleotide Transformer - TATA", "Nucleotide Transformer - Fake TATA", "Nucleotide Transformer - Enhancers",
                                               "scGPT - Multiple Sclerosis", "scGPT - Pancreas"))) %>% 
  dplyr::select(label, pt, ri, distance_diag, head_name, model_name) %>% 
  dplyr::group_by(model_name, label) %>% 
  dplyr::mutate(sig = ifelse(distance_diag > (mean(distance_diag)+sd(distance_diag*3)), "Significantly changed", "No change")) %>% 
  dplyr::ungroup()

gg_zscore_pt_ri_plotlist <- gg_zscore_pt_ri_df %>% 
  dplyr::group_by(model_name) %>% 
  group_map(\(tb.f, key) {
    xy.limits <- range(c(tb.f$ri, tb.f$pt))
    gg_tmp <- tb.f %>% 
      ggplot(aes(x = ri, y = pt)) +
      facet_wrap(~label, ncol = 2) +
      ggrastr::geom_point_rast(aes(alpha = sig, color = sig)) +
      geom_hline(aes(yintercept = 0), linetype = "dashed") +
      geom_vline(aes(xintercept = 0), linetype = "dashed") +
      geom_abline() +
      ggrepel::geom_label_repel(data = . %>% 
                                  dplyr::filter(sig == "Significantly changed") %>%
                                  dplyr::group_by(label) %>% 
                                  dplyr::slice_max(order_by = distance_diag, n = 3),
                                aes(label = head_name),
                                box.padding = 0.5,
                                alpha = 0.9,
                                size = 2,
                                force = 2,
                                seed = 4321) +
      scale_x_continuous(limits = xy.limits) +
      scale_y_continuous(limits = xy.limits) +
      scale_color_manual(values = c("Significantly changed" = "#bc4749", "No change" = "black")) +
      scale_alpha_manual(values = c("Significantly changed" = 1, "No change" = 0.7)) +
      labs(title = key$model_name,
           x = "Random initialized Z-Scores",
           y = "Pre-trained Z-Scores") +
      theme_minimal() +
      theme(plot.title = element_text(size = 10, hjust = 0.5),
            panel.border = element_rect(fill = NA),
            strip.text = element_text(size = 9),
            axis.title = element_text(size = 9),
            axis.text = element_text(size = 8),
            legend.title = element_blank(),
            legend.position = "top")
  })

gg_supS2 <- ggpubr::ggarrange(gg_zscore_pt_ri_plotlist[[1]], gg_zscore_pt_ri_plotlist[[2]],
                              gg_zscore_pt_ri_plotlist[[3]], NULL,
                              gg_zscore_pt_ri_plotlist[[4]], gg_zscore_pt_ri_plotlist[[5]],
                              gg_zscore_pt_ri_plotlist[[6]], NULL,
                              gg_zscore_pt_ri_plotlist[[7]], gg_zscore_pt_ri_plotlist[[8]],
                              labels = c("A", "", "", "", "B", "", "", "", "C", ""),
                              ncol = 2, nrow = 5,
                              common.legend = TRUE,
                              legend = "top")
ggsave('SuppFigureS2_A4.pdf',
       plot = gg_supS2,
       width = 8.3,  # in inches
       height = 11.7, # in inches
       units = "in",
       dpi = 300,
       bg = "white")

## Supp Figure S3 ----
### RPT vs RI ----
gg_zscore_rpt_ri_df <- zscore_df %>% 
  dplyr::filter(training %in% c("rpt", "ri") & model == "dnabert") %>% 
  tidyr::pivot_wider(names_from = training,
                     values_from = zscore) %>% 
  dplyr::mutate(distance_diag = abs(rpt - ri)/sqrt(2),
                label = factor(label,
                               levels = c("0", "1"),
                               labels = c("Negative controls", "Positive controls")),
                head_name = glue::glue("{layer}-{feature}"),
                model_name = glue::glue("{model}_{dataset}"),
                model_name = factor(model_name,
                                    levels = c("dnabert_tata", "dnabert_fake", "dnabert_enhancer"),
                                    labels = c("DNABERT - TATA", "DNABERT - Fake TATA", "DNABERT - Enhancers"))) %>% 
  dplyr::select(label, rpt, ri, distance_diag, head_name, model_name) %>% 
  dplyr::group_by(model_name, label) %>% 
  dplyr::mutate(sig = ifelse(distance_diag > (mean(distance_diag)+sd(distance_diag*3)), "Significantly changed", "No change")) %>% 
  dplyr::ungroup()

gg_zscore_rpt_ri_plotlist <- gg_zscore_rpt_ri_df %>% 
  dplyr::group_by(model_name) %>% 
  group_map(\(tb.f, key) {
    xy.limits <- range(c(tb.f$ri, tb.f$rpt))
    gg_tmp <- tb.f %>% 
      ggplot(aes(x = ri, y = rpt)) +
      facet_wrap(~label, ncol = 2) +
      ggrastr::geom_point_rast(aes(alpha = sig, color = sig)) +
      geom_hline(aes(yintercept = 0), linetype = "dashed") +
      geom_vline(aes(xintercept = 0), linetype = "dashed") +
      geom_abline() +
      ggrepel::geom_label_repel(data = . %>% 
                                  dplyr::filter(sig == "Significantly changed") %>% 
                                  dplyr::group_by(label) %>% 
                                  dplyr::slice_max(order_by = distance_diag, n = 3),
                                aes(label = head_name),
                                box.padding = 0.5,
                                alpha = 0.9,
                                size = 2,
                                force = 3,
                                seed = 4321) +
      scale_x_continuous(limits = xy.limits) +
      scale_y_continuous(limits = xy.limits) +
      scale_color_manual(values = c("Significantly changed" = "#bc4749", "No change" = "black")) +
      scale_alpha_manual(values = c("Significantly changed" = 1, "No change" = 0.7)) +
      labs(title = key$model_name,
           x = "Random initialized Z-Scores",
           y = "Random pre-trained\nZ-Scores") +
      theme_minimal() +
      theme(plot.title = element_text(size = 10, hjust = 0.5),
            panel.border = element_rect(fill = NA),
            strip.text = element_text(size = 9),
            axis.title = element_text(size = 9),
            axis.text = element_text(size = 8),
            legend.title = element_blank(),
            legend.position = "top")
  })

### RPT vs PT ----
gg_zscore_rpt_pt_df <- zscore_df %>% 
  dplyr::filter(training %in% c("rpt", "pt") & model == "dnabert") %>% 
  tidyr::pivot_wider(names_from = training,
                     values_from = zscore) %>% 
  dplyr::mutate(distance_diag = abs(rpt - pt)/sqrt(2),
                label = factor(label,
                               levels = c("0", "1"),
                               labels = c("Negative controls", "Positive controls")),
                head_name = glue::glue("{layer}-{feature}"),
                model_name = glue::glue("{model}_{dataset}"),
                model_name = factor(model_name,
                                    levels = c("dnabert_tata", "dnabert_fake", "dnabert_enhancer"),
                                    labels = c("DNABERT - TATA", "DNABERT - Fake TATA", "DNABERT - Enhancers"))) %>% 
  dplyr::select(label, rpt, pt, distance_diag, head_name, model_name) %>% 
  dplyr::group_by(model_name, label) %>% 
  dplyr::mutate(sig = ifelse(distance_diag > (mean(distance_diag)+sd(distance_diag*3)), "Significantly changed", "No change")) %>% 
  dplyr::ungroup()

gg_zscore_rpt_pt_plotlist <- gg_zscore_rpt_pt_df %>% 
  dplyr::group_by(model_name) %>% 
  group_map(\(tb.f, key) {
    xy.limits <- range(c(tb.f$pt, tb.f$rpt))
    gg_tmp <- tb.f %>% 
      ggplot(aes(x = pt, y = rpt)) +
      facet_wrap(~label, ncol = 2) +
      ggrastr::geom_point_rast(aes(alpha = sig, color = sig)) +
      geom_hline(aes(yintercept = 0), linetype = "dashed") +
      geom_vline(aes(xintercept = 0), linetype = "dashed") +
      geom_abline() +
      ggrepel::geom_label_repel(data = . %>% 
                                  dplyr::filter(sig == "Significantly changed") %>% 
                                  dplyr::group_by(label) %>% 
                                  dplyr::slice_max(order_by = distance_diag, n = 3),
                                aes(label = head_name),
                                box.padding = 0.5,
                                alpha = 0.9,
                                size = 2,
                                force = 3,
                                seed = 4321) +
      scale_x_continuous(limits = xy.limits) +
      scale_y_continuous(limits = xy.limits) +
      scale_color_manual(values = c("Significantly changed" = "#bc4749", "No change" = "black")) +
      scale_alpha_manual(values = c("Significantly changed" = 1, "No change" = 0.7)) +
      labs(title = key$model_name,
           x = "Pre-trained Z-Scores",
           y = "Random pre-trained\nZ-Scores") +
      theme_minimal() +
      theme(plot.title = element_text(size = 10, hjust = 0.5),
            panel.border = element_rect(fill = NA),
            strip.text = element_text(size = 9),
            axis.title = element_text(size = 9),
            axis.text = element_text(size = 8),
            legend.title = element_blank(),
            legend.position = "top")
  })

### RPT vs FT ----
gg_zscore_rpt_ft_df <- zscore_df %>% 
  dplyr::filter(training %in% c("rpt", "ft") & model == "dnabert") %>% 
  tidyr::pivot_wider(names_from = training,
                     values_from = zscore) %>% 
  dplyr::mutate(distance_diag = abs(rpt - ft)/sqrt(2),
                label = factor(label,
                               levels = c("0", "1"),
                               labels = c("Negative controls", "Positive controls")),
                head_name = glue::glue("{layer}-{feature}"),
                model_name = glue::glue("{model}_{dataset}"),
                model_name = factor(model_name,
                                    levels = c("dnabert_tata", "dnabert_fake", "dnabert_enhancer"),
                                    labels = c("DNABERT - TATA", "DNABERT - Fake TATA", "DNABERT - Enhancers"))) %>% 
  dplyr::select(label, rpt, ft, distance_diag, head_name, model_name) %>% 
  dplyr::group_by(model_name, label) %>% 
  dplyr::mutate(sig = ifelse(distance_diag > (mean(distance_diag)+sd(distance_diag*3)), "Significantly changed", "No change")) %>% 
  dplyr::ungroup()

gg_zscore_rpt_ft_plotlist <- gg_zscore_rpt_ft_df %>% 
  dplyr::group_by(model_name) %>% 
  group_map(\(tb.f, key) {
    xy.limits <- range(c(tb.f$ft, tb.f$rpt))
    gg_tmp <- tb.f %>% 
      ggplot(aes(x = ft, y = rpt)) +
      facet_wrap(~label, ncol = 2) +
      ggrastr::geom_point_rast(aes(alpha = sig, color = sig)) +
      geom_hline(aes(yintercept = 0), linetype = "dashed") +
      geom_vline(aes(xintercept = 0), linetype = "dashed") +
      geom_abline() +
      ggrepel::geom_label_repel(data = . %>% 
                                  dplyr::filter(sig == "Significantly changed") %>% 
                                  dplyr::group_by(label) %>% 
                                  dplyr::slice_max(order_by = distance_diag, n = 3),
                                aes(label = head_name),
                                box.padding = 0.5,
                                alpha = 0.9,
                                size = 2,
                                force = 3,
                                seed = 4321) +
      scale_x_continuous(limits = xy.limits) +
      scale_y_continuous(limits = xy.limits) +
      scale_color_manual(values = c("Significantly changed" = "#bc4749", "No change" = "black")) +
      scale_alpha_manual(values = c("Significantly changed" = 1, "No change" = 0.7)) +
      labs(title = key$model_name,
           x = "Fine-tuned Z-Scores",
           y = "Random pre-trained\nZ-Scores") +
      theme_minimal() +
      theme(plot.title = element_text(size = 10, hjust = 0.5),
            panel.border = element_rect(fill = NA),
            strip.text = element_text(size = 9),
            axis.title = element_text(size = 9),
            axis.text = element_text(size = 8),
            legend.title = element_blank(),
            legend.position = "top")
  })

gg_supS3 <- ggpubr::ggarrange(gg_zscore_rpt_ri_plotlist[[1]], gg_zscore_rpt_ri_plotlist[[2]],
                              gg_zscore_rpt_ri_plotlist[[3]], NULL,
                              gg_zscore_rpt_pt_plotlist[[1]], gg_zscore_rpt_pt_plotlist[[2]],
                              gg_zscore_rpt_pt_plotlist[[3]], NULL,
                              gg_zscore_rpt_ft_plotlist[[1]], gg_zscore_rpt_ft_plotlist[[2]],
                              gg_zscore_rpt_ft_plotlist[[3]], NULL,
                              labels = c("A", "", "", "", "B", "", "", "", "C", "", "", ""),
                              ncol = 2, nrow = 6,
                              common.legend = TRUE,
                              legend = "top")
ggsave('SuppFigureS3_A4.pdf',
       plot = gg_supS3,
       width = 8.3,  # in inches
       height = 11.7, # in inches
       units = "in",
       dpi = 300,
       bg = "white")

## Supp Figure S4 ----
gg_zscore_pt_ft_df <- zscore_df %>% 
  dplyr::filter(training %in% c("pt", "ft")) %>% 
  tidyr::pivot_wider(names_from = training,
                     values_from = zscore) %>% 
  dplyr::mutate(distance_diag = abs(pt - ft)/sqrt(2),
                label = factor(label,
                               levels = c("0", "1", "global"),
                               labels = c("Negative controls", "Positive controls", "Global")),
                head_name = glue::glue("{layer}-{feature}"),
                model_name = glue::glue("{model}_{dataset}"),
                model_name = factor(model_name,
                                    levels = c("dnabert_tata", "dnabert_fake", "dnabert_enhancer",
                                               "nt_tata", "nt_fake", "nt_enhancer",
                                               "scgpt_ms", "scgpt_pancreas"),
                                    labels = c("DNABERT - TATA", "DNABERT - Fake TATA", "DNABERT - Enhancers",
                                               "Nucleotide Transformer - TATA", "Nucleotide Transformer - Fake TATA", "Nucleotide Transformer - Enhancers",
                                               "scGPT - Multiple Sclerosis", "scGPT - Pancreas"))) %>% 
  dplyr::select(label, pt, ft, distance_diag, head_name, model_name) %>% 
  dplyr::group_by(model_name, label) %>% 
  dplyr::mutate(sig = ifelse(distance_diag > (mean(distance_diag)+sd(distance_diag*3)), "Significantly changed", "No change")) %>% 
  dplyr::ungroup()

gg_zscore_pt_ft_plotlist <- gg_zscore_pt_ft_df %>% 
  dplyr::group_by(model_name) %>% 
  group_map(\(tb.f, key) {
    xy.limits <- range(c(tb.f$ft, tb.f$pt))
    gg_tmp <- tb.f %>% 
      ggplot(aes(x = ft, y = pt)) +
      facet_wrap(~label, ncol = 2) +
      ggrastr::geom_point_rast(aes(alpha = sig, color = sig)) +
      geom_hline(aes(yintercept = 0), linetype = "dashed") +
      geom_vline(aes(xintercept = 0), linetype = "dashed") +
      geom_abline() +
      ggrepel::geom_label_repel(data = . %>% 
                                  dplyr::filter(sig == "Significantly changed") %>% 
                                  dplyr::group_by(label) %>% 
                                  dplyr::slice_max(order_by = distance_diag, n = 3),
                                aes(label = head_name),
                                box.padding = 0.5,
                                alpha = 0.9,
                                size = 2,
                                force = 3,
                                seed = 4321) +
      scale_x_continuous(limits = xy.limits) +
      scale_y_continuous(limits = xy.limits) +
      scale_color_manual(values = c("Significantly changed" = "#bc4749", "No change" = "black")) +
      scale_alpha_manual(values = c("Significantly changed" = 1, "No change" = 0.7)) +
      labs(title = key$model_name,
           x = "Fine-tuned Z-Scores",
           y = "Pre-trained Z-Scores") +
      theme_minimal() +
      theme(plot.title = element_text(size = 10, hjust = 0.5),
            panel.border = element_rect(fill = NA),
            strip.text = element_text(size = 9),
            axis.title = element_text(size = 9),
            axis.text = element_text(size = 8),
            legend.title = element_blank(),
            legend.position = "top")
  })

gg_supS4 <- ggpubr::ggarrange(gg_zscore_pt_ft_plotlist[[1]], gg_zscore_pt_ft_plotlist[[2]],
                              gg_zscore_pt_ft_plotlist[[3]], NULL,
                              gg_zscore_pt_ft_plotlist[[4]], gg_zscore_pt_ft_plotlist[[5]],
                              gg_zscore_pt_ft_plotlist[[6]], NULL,
                              gg_zscore_pt_ft_plotlist[[7]], gg_zscore_pt_ft_plotlist[[8]],
                              labels = c("A", "", "", "", "B", "", "", "", "C", ""),
                              ncol = 2, nrow = 5,
                              common.legend = TRUE,
                              legend = "top")
ggsave('SuppFigureS4_A4.pdf',
       plot = gg_supS4,
       width = 8.3,  # in inches
       height = 11.7, # in inches
       units = "in",
       dpi = 300,
       bg = "white")


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# ROC metrics ----
## Config ----
open_ablation_scores <- function(files.f){
  names(files.f) <- basename(files.f)
  set_names(names(files.f)) %>% 
    map(\(x) {read_delim(files.f[[x]], delim = "\t", show_col_types = FALSE)}) %>% 
    list_rbind(names_to = "experiment") %>% 
    dplyr::mutate(model = str_split_i(experiment, "_", 2),
                  percent = str_split_i(experiment, "_", 4),
                  model = ifelse(is.na(percent),
                                 model,
                                 glue::glue("{model}_{percent}")),
                  true_label = factor(true_label,
                                      levels = c(0, 1))) %>% 
    dplyr::select(-experiment, -sequence, -percent) %>% 
    dplyr::group_by(model) %>% 
    yardstick::roc_curve(truth = true_label, prob_positive, event_level = "second") %>% 
    dplyr::ungroup() %>% 
    dplyr::mutate(model = factor(model,
                                 levels = c("baseline",
                                            "important_5percent", "important_10percent", "important_20percent", "important_30percent", "important_40percent", "important_50percent",
                                            "unimportant_5percent", "unimportant_10percent", "unimportant_20percent", "unimportant_30percent", "unimportant_40percent", "unimportant_50percent"),
                                 labels = ablation_names))
}
open_ablation_scores_scgpt <- function(files.f, dataset.f){
  wider_delim_names <- if (dataset.f == "ms") {
    paste0("x", 0:17)
  } else {
    paste0("x", 0:13)
  }
  
  names(files.f) <- basename(files.f)
  tmp <- set_names(names(files.f)) %>% 
    map(\(x) {read_delim(files.f[[x]], delim = "\t", show_col_types = FALSE)}) %>% 
    list_rbind(names_to = "experiment") %>% 
    dplyr::mutate(model = str_split_i(experiment, "_", 2),
                  percent = str_split_i(experiment, "_", 4),
                  model = ifelse(is.na(percent),
                                 model,
                                 glue::glue("{model}_{percent}")),
                  prob_values = str_remove_all(prob_values, "\\[|\\]")) %>% 
    dplyr::select(-experiment, -tokens, -percent) %>% 
    dplyr::arrange(true_label) %>% 
    dplyr::mutate(true_label = as_factor(paste0("x", true_label))) %>% 
    tidyr::separate_wider_delim(prob_values,
                                delim = ",",
                                names = wider_delim_names) %>% 
    dplyr::mutate(across(starts_with("x"), ~ as.numeric(.x))) %>% 
    dplyr::group_by(model)
  tmp %>% 
    yardstick::roc_curve(truth = true_label,
                         any_of(unique(tmp$true_label))) %>% 
    dplyr::group_by(model, .level) %>% 
    dplyr::mutate(nrow = row_number()) %>% 
    dplyr::group_by(model, nrow) %>% 
    dplyr::summarise(specificity = mean(specificity),
                     sensitivity = mean(sensitivity)) %>% 
    dplyr::ungroup() %>% 
    dplyr::mutate(model = factor(model,
                                 levels = c("baseline",
                                            "important_5percent", "important_10percent", "important_20percent", "important_30percent", "important_40percent", "important_50percent",
                                            "unimportant_5percent", "unimportant_10percent", "unimportant_20percent", "unimportant_30percent", "unimportant_40percent", "unimportant_50percent"),
                                 labels = ablation_names))
}
### Names
ablation_names <- read_delim('attentions/DNABERT/ablations/ablation_results/TATA-kmer/TATA-kmer_ablation_summary.tsv', delim = "\t", show_col_types = FALSE) %>% 
  dplyr::mutate(Model = str_replace(Model, " Ablated", ""),
                Model = str_replace(Model, "TATA-kmer ", "")) %>% 
  dplyr::pull(Model)
### Color
color <- c("black", 
           "#99e2b4", "#78c6a3", "#56ab91", "#358f80", "#14746f", "#036666",
           "#e01e37", "#c71f37", "#bd1f36", "#a71e34", "#85182a", "#641220")
names(color) <- ablation_names
### Linetype
linetype <- c(1, rep(1,6), rep(2,6))
names(linetype) <- ablation_names
### Linewidth
linewidth <- c(1.5, rep(1,6), rep(1,6))
names(linewidth) <- ablation_names

## Supp Figure S5 ----
### DNABERT - Enhancer ----
dnabert_enhancers_ablation_files <- list.files('attentions/DNABERT/ablations/ablation_results/enhancer',
                                     pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$",
                                     full.names = TRUE)
dnabert_enhancers_ablation_roc <- open_ablation_scores(dnabert_enhancers_ablation_files)
gg_dnabert_enhancers_gc_ablation <- dnabert_enhancers_ablation_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "DNABERT - Enhancers",
       subtitle = "feature: GC",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

### NT - Enhancer ----
nt_enhancers_ablation_files <- list.files('attentions/nucleotide-transformer/ablations/ablation_results/enhancer',
                                               pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$",
                                               full.names = TRUE)
nt_enhancers_ablation_roc <- open_ablation_scores(nt_enhancers_ablation_files[2:length(nt_enhancers_ablation_files)])
gg_nt_enhancers_gc_ablation <- nt_enhancers_ablation_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "Nucleotide Transformer - Enhancers",
       subtitle = "feature: GC",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

### scGPT - MS ----
scgpt_ms_ablation_files <- list.files('attentions/scGPT/ablations/ablation_results/scgpt_ms/GOCC synapse',
                                          pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$",
                                          full.names = TRUE)
scgpt_ms_ablation_roc <- open_ablation_scores_scgpt(scgpt_ms_ablation_files, "ms")
gg_scgpt_ms_ablation <- scgpt_ms_ablation_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "scGPT - Multiple Sclerosis",
       subtitle = "feature: GOCC Synapse",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

### scGPT - Pancreas ----
scgpt_pancreas_ablation_files <- list.files('attentions/scGPT/ablations/ablation_results/scgpt_pancreas/pancreas ductal cell',
                                      pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$",
                                      full.names = TRUE)
scgpt_pancreas_ablation_roc <- open_ablation_scores_scgpt(scgpt_pancreas_ablation_files, "pancreas")
gg_scgpt_pancreas_ablation <- scgpt_pancreas_ablation_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "scGPT - Pancreas",
       subtitle = "feature: Pancreas ductal cell",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

gg_supS5 <- ggpubr::ggarrange(gg_dnabert_enhancers_gc_ablation, gg_nt_enhancers_gc_ablation,
                              gg_scgpt_ms_ablation, gg_scgpt_pancreas_ablation,
                              NULL, NULL,
                              labels = c("A", "", "B", ""),
                              ncol = 2, nrow = 3,
                              common.legend = TRUE,
                              heights = c(0.45, 0.45, 0.1),
                              legend = "bottom")
ggsave('SuppFigureS5_A4.pdf',
       plot = gg_supS5,
       width = 8.3,  # in inches
       height = 11.7, # in inches
       units = "in",
       dpi = 300,
       bg = "white")

## Supp Figure S6 ----
### TATA-kmer ----
dnabert_ablation_files <- list.files('attentions/DNABERT/ablations/ablation_results/TATA-kmer',
                                     pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$",
                                     full.names = TRUE)

dnabert_ablation_roc <- open_ablation_scores(dnabert_ablation_files)
gg_tata_global_ablation <- dnabert_ablation_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "DNABERT - TATA",
       subtitle = "feature: TATAAA k-mer",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

### TATA-kmer positive ----
dnabert_ablation_files_positive <- list.files('attentions/DNABERT/ablations/ablation_results/TATA-kmer-positive',
                                     pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$",
                                     full.names = TRUE)
dnabert_ablation_positive_roc <- open_ablation_scores(dnabert_ablation_files_positive)
gg_tata_positive_ablation <- dnabert_ablation_positive_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "DNABERT - TATA",
       subtitle = "feature: Positive correlated with TATAAA k-mer",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

### TATA-kmer negative ----
dnabert_ablation_files_negative <- list.files('attentions/DNABERT/ablations/ablation_results/TATA-kmer-negative',
                                              pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$",
                                              full.names = TRUE)
dnabert_ablation_negative_roc <- open_ablation_scores(dnabert_ablation_files_negative)

gg_tata_negative_ablation <- dnabert_ablation_negative_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "DNABERT - TATA",
       subtitle = "feature: Negative correlated with TATAAA k-mer",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

gg_supS6 <- ggpubr::ggarrange(gg_tata_global_ablation, gg_tata_positive_ablation,
                              gg_tata_negative_ablation, NULL,
                              NULL, NULL,
                              labels = c("A", "B", "C", "", "", ""),
                              ncol = 2, nrow = 3,
                              common.legend = TRUE,
                              legend = "right")
ggsave('SuppFigureS6_A4_raw.pdf',
       plot = gg_supS6,
       width = 8.3,  # in inches
       height = 11.7, # in inches
       units = "in",
       dpi = 300,
       bg = "white")

## Supp Figure S7 ----
### DNABERT - Fake TATA GC ----
dnabert_fake_tata_gc_ablation_files <- list.files('attentions/DNABERT/ablations/ablation_results/fake_tata',
                                               pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$",
                                               full.names = TRUE)
dnabert_fake_tata_gc_ablation_roc <- open_ablation_scores(dnabert_fake_tata_gc_ablation_files[2:length(dnabert_fake_tata_gc_ablation_files)])
gg_dnabert_fake_tata_gc_ablation <- dnabert_fake_tata_gc_ablation_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "DNABERT - Fake TATA",
       subtitle = "feature: GC",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

### DNABERT - Fake TATA kmer ----
dnabert_fake_tata_kmer_ablation_files <- list.files('attentions/DNABERT/ablations/ablation_results/fake_TATAkmer',
                                               pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$",
                                               full.names = TRUE)
dnabert_fake_tata_kmer_ablation_roc <- open_ablation_scores(dnabert_fake_tata_kmer_ablation_files)
gg_dnabert_fake_tata_kmer_ablation <- dnabert_fake_tata_kmer_ablation_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "DNABERT - Fake TATA",
       subtitle = "feature: TATAAA k-mer",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

### NT - Fake TATA GC ----
nt_fake_tata_gc_ablation_files <- list.files('attentions/nucleotide-transformer/ablations/ablation_results/fake_tata',
                                          pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$",
                                          full.names = TRUE)
nt_fake_tata_gc_ablation_roc <- open_ablation_scores(nt_fake_tata_gc_ablation_files[2:length(nt_fake_tata_gc_ablation_files)])
gg_nt_fake_tata_gc_ablation <- nt_fake_tata_gc_ablation_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "Nucleotide Transformer - Fake TATA",
       subtitle = "feature: GC",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

### NT - Fake TATA kmer ----
nt_fake_tata_kmer_ablation_files <- list.files('attentions/nucleotide-transformer/ablations/ablation_results/fake_TATA-kmer',
                                             pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$",
                                             full.names = TRUE)
nt_fake_tata_kmer_ablation_roc <- open_ablation_scores(nt_fake_tata_kmer_ablation_files[2:length(nt_fake_tata_kmer_ablation_files)])
gg_nt_fake_tata_kmer_ablation <- nt_fake_tata_kmer_ablation_roc %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "Nucleotide Transformer - Fake TATA",
       subtitle = "feature: TATAAA k-mer",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

### NT - Fake TATA GC window ----
get_tata_windows <- function(tb){
  tb %>% 
    dplyr::filter(layer == "layer0" & head == "head0") %>% 
    dplyr::arrange(layer, head, name, position) %>% 
    dplyr::group_by(name) %>%
    dplyr::mutate(tata_flag = kmers == "ATATAA",
                  tata_grp = ifelse(tata_flag, cumsum(!lag(tata_flag, default = FALSE)), NA_integer_)) %>% 
    dplyr::group_by(name, tata_grp) %>%
    dplyr::mutate(center_pos = if (!all(is.na(tata_grp))) {mean(position[tata_flag])} else { NA_real_ },
                  center_pos = ifelse(position == center_pos, 1, NA_integer_)) %>% 
    dplyr::ungroup() %>% 
    dplyr::group_by(name) %>% 
    dplyr::group_split() %>% 
    map(\(tb1){
      f.keys <- tb1 %>% 
        dplyr::group_by(tata_grp, center_pos) %>% 
        dplyr::group_keys() %>% 
        dplyr::filter(!is.na(center_pos))
      map2(f.keys$tata_grp, f.keys$center_pos, \(f.grp, f.center) {
        tb1 %>% 
          dplyr::mutate(tata_position = row_number()-f.grp,
                        tata_grp = glue::glue("{name}_group_{f.grp}")) %>% 
          dplyr::filter(tata_position %in% seq(-100, 100)) %>% 
          dplyr::select(name, kmers, position, tata_position, tata_grp)
      })
    }) %>% bind_rows()
}

#### Fake TATA sequences
fake_tata_sequences <- readDNAStringSet("sequences/TATA/fake_TATA_test.fa")
fake_tata_sequences <- tibble(tag = str_split_i(names(fake_tata_sequences), "\\|", 2),
                              name = paste0("sequence", seq(1, length(fake_tata_sequences))),
                              sequence = as.character(fake_tata_sequences))

#### Finetuning ----
fake_nt_ft <- pd$read_pickle('files/raw/NT_fake_tata_scores_dictionary_attention.pkl') %>%
  map(bind_cols) %>%
  list_rbind(names_to = "sequence") %>%
  dplyr::group_by(sequence) %>% 
  dplyr::mutate(position = row_number()) %>% 
  dplyr::ungroup() %>% 
  dplyr::left_join(fake_tata_sequences, by = "sequence") %>%
  dplyr::select(name, kmers, position, tag, starts_with('layer')) %>% 
  dplyr::mutate(across(where(~ is.numeric(.x)), ~ map_dbl(.x, ~ .x))) %>% 
  tidyr::pivot_longer(cols = starts_with('layer'),
                      names_to = 'id',
                      values_to = 'attention_scores') %>%
  tidyr::separate_wider_delim(cols = id, delim = "-", names = c("layer", "head"))

fake_nt_ft_tata <- get_tata_windows(fake_nt_ft)

fake_nt_ft_tata_tb <- fake_nt_ft %>% 
  dplyr::left_join(fake_nt_ft_tata,
                   by = c("name", "kmers", "position"),
                   relationship = "many-to-many") %>% 
  dplyr::filter(!is.na(tata_grp)) %>% 
  dplyr::mutate(model = "ft")

#### Pretrained ----
fake_nt_pt <- pd$read_pickle('files/raw/NT_fake_tata_scores_dictionary_pretrained_attention.pkl') %>%
  map(bind_cols) %>%
  list_rbind(names_to = "sequence") %>%
  dplyr::group_by(sequence) %>% 
  dplyr::mutate(position = row_number()) %>% 
  dplyr::ungroup() %>% 
  dplyr::left_join(fake_tata_sequences, by = "sequence") %>%
  dplyr::select(name, kmers, position, tag, starts_with('layer')) %>% 
  dplyr::mutate(across(where(~ is.numeric(.x)), ~ map_dbl(.x, ~ .x))) %>% 
  tidyr::pivot_longer(cols = starts_with('layer'),
                      names_to = 'id',
                      values_to = 'attention_scores') %>%
  tidyr::separate_wider_delim(cols = id, delim = "-", names = c("layer", "head"))

fake_nt_pt_tata <- get_tata_windows(fake_nt_pt)

fake_nt_pt_tata_tb <- fake_nt_pt %>% 
  dplyr::left_join(fake_nt_pt_tata,
                   by = c("name", "kmers", "position"),
                   relationship = "many-to-many") %>% 
  dplyr::filter(!is.na(tata_grp)) %>% 
  dplyr::mutate(model = "pt")

#### Random init ----
fake_nt_ri <- pd$read_pickle('files/raw/NT_fake_tata_scores_dictionary_random_init_attention.pkl') %>%
  map(bind_cols) %>%
  list_rbind(names_to = "sequence") %>%
  dplyr::group_by(sequence) %>% 
  dplyr::mutate(position = row_number()) %>% 
  dplyr::ungroup() %>% 
  dplyr::left_join(fake_tata_sequences, by = "sequence") %>%
  dplyr::select(name, kmers, position, tag, starts_with('layer')) %>% 
  dplyr::mutate(across(where(~ is.numeric(.x)), ~ map_dbl(.x, ~ .x))) %>% 
  tidyr::pivot_longer(cols = starts_with('layer'),
                      names_to = 'id',
                      values_to = 'attention_scores') %>%
  tidyr::separate_wider_delim(cols = id, delim = "-", names = c("layer", "head"))

fake_nt_ri_tata <- get_tata_windows(fake_nt_ri)

fake_nt_ri_tata_tb <- fake_nt_ri %>% 
  dplyr::left_join(fake_nt_ri_tata,
                   by = c("name", "kmers", "position"),
                   relationship = "many-to-many") %>% 
  dplyr::filter(!is.na(tata_grp)) %>% 
  dplyr::mutate(model = "ri")

#### Plot
nt_fake_tata_gc_window_df <- bind_rows(fake_nt_ft_tata_tb, fake_nt_pt_tata_tb, fake_nt_ri_tata_tb)  %>% 
  dplyr::filter(layer == "layer24" & head == "head9" & tag == 1) %>%
  dplyr::mutate(gc = str_count(kmers, "[GCgc]") / str_length(kmers)) %>% 
  dplyr::group_by(model, tata_position) %>% 
  dplyr::summarise(attention_scores = mean(attention_scores), 
                   gc = mean(gc),
                   .groups = "drop") %>% 
  dplyr::mutate(model = factor(model,
                               levels = c("pt", "ft", "ri"),
                               labels = c("Pre-trained", "Fine-tuned", "Random initialized")))

gg_nt_fake_tata_gc_windowA <- nt_fake_tata_gc_window_df %>% 
  ggplot(aes(x = tata_position, y = attention_scores, color = model)) +
  geom_line(linewidth = 1) +
  labs(title = "Nucleotide Transformer - Fake TATA",
       subtitle = "layer24-head9",
       y = "Mean attention\nscore",
       color = "") +
  scale_color_manual(values = c("Pre-trained" = "#e07a5f", "Fine-tuned" = "#81b29a", "Random initialized" = "#5B6386", "%GC" = "black")) +
  theme_bw() +
  theme(plot.title = element_text(size = 13, hjust = 0.5),
        plot.subtitle = element_text(size = 13, hjust = 0.5),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 12),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 11),
        axis.ticks.x = element_blank(),
        legend.text = element_text(size = 10),
        legend.position = "bottom")
gg_nt_fake_tata_gc_windowB <- nt_fake_tata_gc_window_df %>% 
  dplyr::filter(model == "Fine-tuned") %>% 
  ggplot(aes(x = tata_position, y = gc)) +
  geom_line(linewidth = 1, linetype = "dashed") +
  labs(x = "TATAAA k-mer position (bp)",
       y = "GC content\n(%)",
       color = "") +
  ylim(c(0,1)) +
  theme_bw() +
  theme(panel.grid.minor = element_blank(), 
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 11),
        legend.text = element_text(size = 10),
        legend.position = "right")

gg_supS7AB <- ggpubr::ggarrange(gg_dnabert_fake_tata_gc_ablation, gg_dnabert_fake_tata_kmer_ablation,
                              gg_nt_fake_tata_gc_ablation, gg_nt_fake_tata_kmer_ablation,
                              labels = c("A", "", "B"),
                              ncol = 2, nrow = 2,
                              common.legend = TRUE,
                              legend = "bottom")
gg_supS7C <- ggpubr::ggarrange(gg_nt_fake_tata_gc_windowA, gg_nt_fake_tata_gc_windowB,
                               ncol = 1,
                               heights = c(0.6, 0.4),
                               common.legend = TRUE,
                               legend = "bottom")
gg_supS7 <- ggpubr::ggarrange(gg_supS7AB, gg_supS7C,
                              labels = c("", "C"),
                              ncol = 1, nrow = 2,
                              heights = c(0.7, 0.3))
ggsave('SuppFigureS7_A4.pdf',
       plot = gg_supS7,
       width = 8.3,  # in inches
       height = 11.7, # in inches
       units = "in",
       dpi = 300,
       bg = "white")
