library(qs)
library(tidyverse)
library(Biostrings)
library(yardstick)
library(ggnewscale)
library(ggpmisc)
library(ggpointdensity)
library(ggpubr)
library(ggsci)
library(ggseqlogo)
library(RColorBrewer)
library(viridis)
library(reticulate)

options(dplyr.summarise.inform = FALSE)


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Options ----
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


# Files ----
## sequences ----
tata_sequences <- readDNAStringSet("attentions/test_datasets/custom_TATA_test.fa")
tata_sequences <- tibble(tag = str_split_i(names(tata_sequences), "\\|", 2),
                         name = paste0("sequence", seq(1, length(tata_sequences))),
                         sequence = as.character(tata_sequences))

## names ----
tata_files <- c("dnabert_ft" = "files/processed/DNABERT_custom_tata_scores_dictionary_attention_annotations.qs",
                "dnabert_pt" = "files/processed/DNABERT_custom_tata_scores_dictionary_attention_pretrained_annotations.qs",
                "dnabert_rpt" = "files/processed/DNABERT_custom_tata_scores_dictionary_attention_random_pretrained_annotations.qs",
                "dnabert_ri" = "files/processed/DNABERT_custom_tata_scores_dictionary_attention_random_init_annotations.qs",
                "nt_ft" = "files/processed/NT_custom_tata_scores_dictionary_attention_annotations.qs",
                "nt_pt" = "files/processed/NT_custom_tata_scores_dictionary_attention_pretrained_annotations.qs",
                "nt_ri" = "files/processed/NT_custom_tata_scores_dictionary_attention_random_init_annotations.qs")

scgpt_files <- c("ms_ft" = "files/processed/scgpt_ms_finetuned_scores.qs",
                 "ms_pt" = "files/processed/scgpt_ms_pretrained_scores.qs",
                 "ms_ri" = "files/processed/scgpt_ms_random_init_scores.qs")

tata_zscore_files <- c("dnabert_ft_0" = "files/z_score_matrices/DNABERT_TATA/finetuned/label_specific/0_centered_z_scores.csv",
                       "dnabert_ft_1" = "files/z_score_matrices/DNABERT_TATA/finetuned/label_specific/1_centered_z_scores.csv",
                       "dnabert_pt_0" = "files/z_score_matrices/DNABERT_TATA/pretrained/label_specific/0_centered_z_scores.csv",
                       "dnabert_pt_1" = "files/z_score_matrices/DNABERT_TATA/pretrained/label_specific/1_centered_z_scores.csv",
                       "dnabert_ri_0" = "files/z_score_matrices/DNABERT_TATA/random_init/label_specific/0_centered_z_scores.csv",
                       "dnabert_ri_1" = "files/z_score_matrices/DNABERT_TATA/random_init/label_specific/1_centered_z_scores.csv")

scgpt_zscore_files <- c("scgpt_ft_0" = "files/z_score_matrices/scgpt_ms/finetuned/label_specific/13_centered_z_scores.csv", #oligodendrocyte A n=154
                        "scgpt_ft_1" = "files/z_score_matrices/scgpt_ms/finetuned/label_specific/4_centered_z_scores.csv", #astrocyte n=107
                        "scgpt_pt_0" = "files/z_score_matrices/scgpt_ms/pretrained/label_specific/13_centered_z_scores.csv",
                        "scgpt_pt_1" = "files/z_score_matrices/scgpt_ms/pretrained/label_specific/4_centered_z_scores.csv",
                        "scgpt_ri_0" = "files/z_score_matrices/scgpt_ms/random_init/label_specific/13_centered_z_scores.csv",
                        "scgpt_ri_1" = "files/z_score_matrices/scgpt_ms/random_init/label_specific/4_centered_z_scores.csv")

## scores ----
tata_df_list <- set_names(names(tata_files)) %>% map(\(x) qread(tata_files[[x]]), .progress = TRUE)

scgpt_df_list <- set_names(names(scgpt_files)) %>% map(\(x) qread(scgpt_files[[x]]), .progress = TRUE)

## layer,head,tag metrics ----
tata_by_layer_head_tag_metrics <- set_names(names(tata_files)) %>% map(\(x) qread(glue::glue('files/intermediate/custom_tata_{x}_layer_head_tag_metrics.qs')), .progress = TRUE)

## z-score matrices ----
tata_zscore_df <- set_names(names(tata_zscore_files)) %>% map(\(x) {
  read_csv(tata_zscore_files[[x]], show_col_types = FALSE) %>% 
    dplyr::rename(layer = `...1`) %>% 
    tidyr::pivot_longer(cols = -layer,
                        names_to = "feature",
                        values_to = "zscore") %>% 
    dplyr::mutate(sequence = str_split_i(x, "_", 3),
                  model = str_split_i(x, "_", 1),
                  training = str_split_i(x, "_", 2))
}) %>% bind_rows()

scgpt_zscore_df <- set_names(names(scgpt_zscore_files)) %>% map(\(x) {
  read_csv(zscore_files_scgpt[[x]], show_col_types = FALSE) %>% 
    dplyr::rename(layer = `...1`) %>% 
    tidyr::pivot_longer(cols = -layer,
                        names_to = "feature",
                        values_to = "zscore") %>% 
    dplyr::mutate(sequence = str_split_i(x, "_", 3),
                  model = str_split_i(x, "_", 1),
                  training = str_split_i(x, "_", 2))
}) %>% bind_rows()

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Context analysis - Figure 2 ----
dnabert_threshold <- mean(tata_by_layer_head_tag_metrics$dnabert_ft$distance_diag)+sd(tata_by_layer_head_tag_metrics$dnabert_ft$distance_diag)*3
manual_explanation_by_layer_head_tag_plot <- function(){
  ### Scatter plot
  gg_scatter <- tata_by_layer_head_tag_metrics$dnabert_ft %>% 
    ggplot(aes(x = positive, y = negative)) +
    geom_point() +
    geom_point(data = . %>% dplyr::filter(distance_diag > dnabert_threshold), color = "#bc4749") +
    geom_abline() +
    geom_hline(aes(yintercept = 0), linetype = "dashed") +
    geom_vline(aes(xintercept = 0), linetype = "dashed") +
    ylim(-1,1) +
    xlim(-1,1) +
    scale_color_viridis_c(option = "G") +
    ggnewscale::new_scale_color() +
    ggrepel::geom_label_repel(data = . %>% dplyr::filter(id %in% c("layer10_head0_TATAAA", "layer11_head10_CCCGAG", "layer6_head5_TTTTTT", "layer11_head2_CTTTAT", "layer11_head5_CTTTAT")),
                              aes(label = id, color = id),
                              min.segment.length = 0.01,
                              seed = 1234,
                              size = 4,
                              label.size = 1.5) +
    ggrepel::geom_label_repel(data = . %>% dplyr::filter(id %in% c("layer10_head0_TATAAA", "layer11_head10_CCCGAG", "layer6_head5_TTTTTT", "layer11_head2_CTTTAT", "layer11_head5_CTTTAT")),
                              aes(label = id),
                              min.segment.length = 0.01,
                              seed = 1234,
                              size = 4,
                              label.size = 0) +
    scale_color_manual(values = c("#0D8B96", "#79CAF6", "#C46E85", "#E0C867", "#F47E3E")) +
    labs(title = "Fine-tuned DNABERT - TATA",
         subtitle = "Difference between k-mer and head mean attention scores\n(k-mer_score - head_score)", 
         x = "Positive controls",
         y = "Negative controls") +
    theme_minimal() +
    theme(plot.title = element_text(size = 15, hjust = 0.5),
          plot.subtitle = element_text(size = 13, hjust = 0.5),
          panel.border = element_rect(fill = NA),
          axis.ticks.y = element_line(),
          axis.title = element_text(size = 13),
          axis.text = element_text(size = 12),
          legend.position = "none")
  
  ### Boxplots
  bxplot_all_df <- tata_df_list$dnabert_ft %>% 
    dplyr::mutate(label = glue::glue("{layer}_{head}")) %>% 
    dplyr::filter(label %in% c("layer10_head0", "layer11_head10", "layer6_head5", "layer11_head2", "layer11_head5")) %>% 
    dplyr::select(label, tag, attention_scores) %>% 
    dplyr::group_by(label, tag) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop") %>% 
    dplyr::mutate(tag = factor(tag,
                               levels = c(0, 1),
                               labels = c("Negative controls", "Positive controls")),
                  label = factor(label,
                                 levels = c("layer6_head5", "layer11_head5", "layer11_head10", "layer11_head2", "layer10_head0"),
                                 labels = c("Lower attention\nNegative = Positive\nlayer6_head5_TTTTTT",
                                            "Downregulation\nin positive controls\nUpregulation\nin negative controls\nlayer11_head5_CTTTAT",
                                            "Downregulation\nin negative controls\nlayer11_head10_CCCGAG",
                                            "Higher attention\nNegative = Positive\nlayer11_head2_CTTTAT",
                                            "Upregulation\nin positive controls\nlayer10_head0_TATAAA")))
  bxplot_kmers_df <- tata_df_list$dnabert_ft %>% 
    dplyr::mutate(label = glue::glue("{layer}_{head}"),
                  select_id = glue::glue("{layer}_{head}_{kmers}")) %>% 
    dplyr::filter(select_id %in% c("layer10_head0_TATAAA", "layer11_head10_CCCGAG", "layer6_head5_TTTTTT", "layer11_head2_CTTTAT", "layer11_head5_CTTTAT")) %>% 
    dplyr::select(label, select_id, tag, attention_scores) %>% 
    dplyr::mutate(tag = factor(tag,
                               levels = c(0, 1),
                               labels = c("Negative controls", "Positive controls")),
                  label = factor(label,
                                 levels = c("layer6_head5", "layer11_head5", "layer11_head10", "layer11_head2", "layer10_head0"),
                                 labels = c("Lower attention\nNegative = Positive\nlayer6_head5_TTTTTT",
                                            "Downregulation\nin positive controls\nUpregulation\nin negative controls\nlayer11_head5_CTTTAT",
                                            "Downregulation\nin negative controls\nlayer11_head10_CCCGAG",
                                            "Higher attention\nNegative = Positive\nlayer11_head2_CTTTAT",
                                            "Upregulation\nin positive controls\nlayer10_head0_TATAAA")))
  gg_box <- bxplot_kmers_df %>% 
    ggplot(aes(x = tag, y = attention_scores, fill = tag)) +
    ggh4x::facet_grid2(~label,
                       strip = ggh4x::strip_themed(background_x = list(element_rect(color = c("#F47E3E"), linewidth = 1.5),
                                                                       element_rect(color = c("#E0C867"), linewidth = 1.5),
                                                                       element_rect(color = c("#79CAF6"), linewidth = 1.5),
                                                                       element_rect(color = c("#C46E85"), linewidth = 1.5),
                                                                       element_rect(color = c("#0D8B96"), linewidth = 1.5)))) +
    geom_boxplot() +
    scale_fill_manual(values = c("#ffddd2", "#83c5be"), name = "") +
    ggnewscale::new_scale_fill() +
    geom_point(data = bxplot_all_df, aes(color = "Mean head\nattention score"), shape = 8, size = 3, stroke = 1) +
    scale_color_manual(values = c("Mean head\nattention score"="#bc4749"), name = "") +
    labs(x = "",
         y = "Attention scores") +
    theme_bw() +
    theme(strip.background = element_blank(), 
          strip.text = element_text(size = 11),
          axis.title = element_text(size = 13),
          axis.ticks.x = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_text(size = 12),
          legend.position = "bottom",
          legend.text = element_text(size = 13))
  
  ### Heatmap
  selected_tata_heads <- tata_by_layer_head_tag_metrics$dnabert_ft %>% 
    dplyr::filter(kmers == "TATAAA") %>% 
    dplyr::slice_max(order_by = distance_diag, n = 5) %>% 
    dplyr::mutate(id = glue::glue("{layer}_{head}")) %>% 
    dplyr::pull(id)
  all_kmers_matrix <- tata_df_list$dnabert_ft %>% 
    dplyr::select(layer, head, tag, attention_scores) %>% 
    dplyr::group_by(layer, head, tag) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  by_tata_matrix <- tata_df_list$dnabert_ft %>% 
    dplyr::filter(kmers == "TATAAA") %>% 
    dplyr::select(layer, head, kmers, tag, attention_scores) %>% 
    dplyr::group_by(layer, head, kmers, tag) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  gg_heatmap <- all_kmers_matrix %>% 
    dplyr::mutate(kmers = "all") %>% 
    bind_rows(by_tata_matrix) %>% 
    dplyr::mutate(tag = factor(tag,
                               levels = c(0, 1),
                               labels = c("Negative controls", "Positive controls")),
                  kmers = factor(kmers,
                                 levels = c("all", "TATAAA"),
                                 labels = c("head attention scores", "TATAAA attention scores")),
                  layer = factor(layer,
                                 levels = dnabert_layer_factors),
                  head = factor(head,
                                levels = dnabert_head_factors),
                  id = glue::glue("{layer}_{head}")) %>% 
    ggplot(.,
           aes(x = head, y = layer, fill = attention_scores)) +
    facet_wrap(~kmers+tag) +
    geom_tile() +
    geom_tile(data = . %>% dplyr::filter(id %in% selected_tata_heads),
              alpha = 0, color = "#bc4749", linewidth = 1) +
    scale_fill_viridis_c(option = 'G', direction = -1, limits = c(0,1)) +
    labs(subtitle = "Fine-tuned DNABERT - TATA",
         fill = "Mean attention\nscore",
         caption = "Red heads correspond to top 5 highlighted TATAAA kmers") +
    theme_minimal() +
    theme(plot.subtitle = element_text(size = 13, hjust = 0.5),
          strip.text = element_text(size = 11),
          axis.title = element_blank(),
          axis.text.x = element_text(angle = 45, size = 10, hjust = 1),
          legend.position = "right",
          legend.text = element_text(size = 12),
          plot.caption = element_text(size = 11))
  
  ### Panel construction
  gg1 <- ggarrange(gg_scatter, gg_heatmap,
                   ncol = 2,
                   widths = c(0.6, 0.5),
                   labels = c("A", "C"))
  ggarrange(gg1, gg_box,
            nrow = 2,
            heights = c(0.6, 0.4),
            labels = c("", "B"))
}

## Plot - Figure 2 ----
explanation_plot <- manual_explanation_by_layer_head_tag_plot()


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Z-score analysis - Figure 3 ----
## A) - PT vs RI ----
gg_zscoreA <- tata_zscore_df %>% 
  dplyr::filter(model == "dnabert" & training %in% c("pt", "ri")) %>% 
  tidyr::pivot_wider(names_from = training,
                     values_from = zscore) %>% 
  dplyr::mutate(distance_diag = abs(pt - ri)/sqrt(2),
                sequence = factor(sequence,
                                  levels = c(0, 1),
                                  labels = c("Negative controls", "Positive controls")),
                label = glue::glue("{layer}-{feature}")) %>% 
  dplyr::group_by(sequence) %>% 
  dplyr::mutate(selected = ifelse(distance_diag > (mean(distance_diag)+sd(distance_diag*3)), "selected", "no_selected")) %>% 
  dplyr::ungroup() %>% 
  ggplot(aes(x = ri, y = pt)) +
  facet_wrap(~sequence) +
  geom_point(data = . %>% dplyr::filter(selected == "no_selected"), alpha = 0.7) +
  geom_point(data = . %>% 
               dplyr::filter(selected == "selected") %>% 
               dplyr::mutate(feature = factor(feature,
                                              levels = c("TSS", "repeat_LTR", "GC", "repeat_SINE", "RXR-like", "repeat_Simple_repeat", "TATAAA", "ATATAA"),
                                              labels = c("TSS", "repeat-LTR", "GC", "repeat-SINE", "RXR-like","repeat-Simple", "TATA-kmer", "TATA-kmer"))),
             aes(color = feature)) +
  geom_hline(aes(yintercept = 0), linetype = "dashed") +
  geom_vline(aes(xintercept = 0), linetype = "dashed") +
  geom_abline() +
  ggrepel::geom_label_repel(data = . %>% 
                              dplyr::filter(sequence == "Negative controls") %>% 
                              dplyr::slice_max(order_by = distance_diag, n = 3),
                            aes(label = label),
                            box.padding = 1,
                            alpha = 0.9,
                            size = 3,
                            force = 2,
                            nudge_y = -20,
                            seed = 1234) +
  ggrepel::geom_label_repel(data = . %>% 
                              dplyr::filter(sequence == "Positive controls") %>% 
                              dplyr::slice_max(order_by = distance_diag, n = 3),
                            aes(label = label),
                            box.padding = 1,
                            alpha = 0.9,
                            size = 3,
                            force = 2,
                            seed = 1234) +
  xlim(c(-80, 55)) +
  ylim(c(-80, 55)) +
  scale_color_manual(values = c("TSS" = "#bc4749", "repeat-LTR" = "#a3b18a", "GC" = "#9f86c0", "repeat-SINE" = "#588157", "RXR-like" = "#f4a261", "repeat-Simple" = "#3a5a40", "TATA-kmer" = "#0096c7")) +
  labs(x = "Random initialized Z-Scores",
       y = "Pre-trained Z-Scores",
       color = "Significantly changed\nfeatures") +
  theme_minimal() +
  theme(plot.title = element_text(size = 15, hjust = 0.5),
        plot.subtitle = element_text(size = 13, hjust = 0.5),
        panel.border = element_rect(fill = NA),
        strip.text = element_text(size = 12),
        axis.ticks = element_line(),
        axis.title = element_text(size = 13),
        axis.text = element_text(size = 12),
        legend.position = "")

## B) - PT vs FT ----
gg_zscoreB <- tata_zscore_df %>% 
  dplyr::filter(model == "dnabert" & training %in% c("pt", "ft")) %>% 
  tidyr::pivot_wider(names_from = training,
                     values_from = zscore) %>% 
  dplyr::mutate(distance_diag = abs(ft - pt)/sqrt(2),
                sequence = factor(sequence,
                                  levels = c(0, 1),
                                  labels = c("Negative controls", "Positive controls")),
                label = glue::glue("{layer}-{feature}")) %>% 
  dplyr::group_by(sequence) %>% 
  dplyr::mutate(selected = ifelse(distance_diag > (mean(distance_diag)+sd(distance_diag*3)), "selected", "no_selected")) %>% 
  dplyr::ungroup() %>% 
  ggplot(aes(x = ft, y = pt)) +
  facet_wrap(~sequence) +
  geom_point(data = . %>% dplyr::filter(selected == "no_selected"), alpha = 0.7) +
  geom_point(data = . %>% 
               dplyr::filter(selected == "selected") %>% 
               dplyr::mutate(feature = factor(feature,
                                              levels = c("TSS", "GC", "RXR-like", "repeat_LTR", "repeat_SINE",  "repeat_Simple_repeat", "TATAAA", "ATATAA"),
                                              labels = c("TSS", "GC", "RXR-like", "repeat-LTR", "repeat-SINE", "repeat-Simple",  "TATA-kmer", "TATA-kmer"))),
             aes(color = feature)) +
  geom_hline(aes(yintercept = 0), linetype = "dashed") +
  geom_vline(aes(xintercept = 0), linetype = "dashed") +
  geom_abline() +
  ggrepel::geom_label_repel(data = . %>% 
                              dplyr::filter(sequence == "Negative controls") %>% 
                              dplyr::slice_max(order_by = distance_diag, n = 3),
                            aes(label = label),
                            box.padding = 1,
                            alpha = 0.9,
                            size = 3,
                            force = 2,
                            seed = 1234) +
  ggrepel::geom_label_repel(data = . %>% 
                              dplyr::filter(sequence == "Positive controls") %>% 
                              dplyr::slice_max(order_by = distance_diag, n = 3),
                            aes(label = label),
                            box.padding = 1,
                            alpha = 0.9,
                            size = 3,
                            force = 2,
                            nudge_y = -10,
                            seed = 1234) +
  xlim(c(-80, 55)) +
  ylim(c(-80, 55)) +
  scale_color_manual(values = c("TSS" = "#bc4749", "repeat-LTR" = "#a3b18a", "GC" = "#9f86c0", "repeat-SINE" = "#588157", "RXR-like" = "#f4a261", "repeat-Simple" = "#3a5a40", "TATA-kmer" = "#0096c7")) +
  labs(x = "Fine-tuned Z-Scores",
       y = "Pre-trained Z-Scores",
       color = "Significantly changed\nfeatures") +
  theme_minimal() +
  theme(plot.title = element_text(size = 15, hjust = 0.5),
        plot.subtitle = element_text(size = 13, hjust = 0.5),
        panel.border = element_rect(fill = NA),
        strip.text = element_text(size = 12),
        axis.ticks = element_line(),
        axis.title = element_text(size = 13),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 13),
        legend.text = element_text(size = 11),
        legend.position = "top")

## C) - TSS feature for TATA ----
get_tss_windows <- function(tb){
  tb %>% 
    dplyr::select(-repeat_Simple_repeat) %>% 
    dplyr::filter(layer == "layer0" & head == "head0") %>% 
    dplyr::arrange(layer, head, name, position) %>% 
    dplyr::group_by(name) %>%
    dplyr::mutate(tss_flag = TSS == 1,
                  tss_grp = ifelse(tss_flag, cumsum(!lag(tss_flag, default = FALSE)), NA_integer_)) %>% 
    dplyr::group_by(name, tss_grp) %>%
    dplyr::mutate(center_pos = if (!all(is.na(tss_grp))) {mean(position[tss_flag])} else { NA_real_ },
                  center_pos = ifelse(position == center_pos, 1, NA_integer_)) %>% 
    dplyr::ungroup() %>% 
    dplyr::group_by(name) %>% 
    dplyr::group_split() %>% 
    map(\(tb1){
      f.keys <- tb1 %>% 
        dplyr::group_by(tss_grp, center_pos) %>% 
        dplyr::group_keys() %>% 
        dplyr::filter(!is.na(center_pos))
      map2(f.keys$tss_grp, f.keys$center_pos, \(f.grp, f.center) {
        tb1 %>% 
          dplyr::mutate(TSS_position = row_number()-f.grp,
                        TSS_grp = glue::glue("{name}_group_{f.grp}")) %>% 
          dplyr::filter(TSS_position %in% seq(-100, 100)) %>% 
          dplyr::select(name, kmers, position, TSS_position, TSS_grp)
      })
    }) %>% bind_rows()
}

### Finetuning
dnabert_ft <- pd$read_pickle('files/raw/DNABERT_custom_tata_scores_dictionary_attention_annotations.pkl') %>%
  map(bind_cols) %>%
  list_rbind(names_to = "sequence") %>%
  dplyr::left_join(tata_sequences, by = "sequence") %>%
  dplyr::select(name, kmers, position, tag, TSS, repeat_Simple_repeat, starts_with('layer')) %>% 
  dplyr::mutate(across(where(~ is.numeric(.x)), ~ map_dbl(.x, ~ .x))) %>% 
  tidyr::pivot_longer(cols = starts_with('layer'),
                      names_to = 'id',
                      values_to = 'attention_scores') %>%
  tidyr::separate_wider_delim(cols = id, delim = "-", names = c("layer", "head"))

dnabert_ft_tss <- get_tss_windows(dnabert_ft)

dnabert_ft_tss_tb <- dnabert_ft %>% 
  dplyr::left_join(dnabert_ft_tss,
                   by = c("name", "kmers", "position"),
                   relationship = "many-to-many") %>% 
  dplyr::filter(!is.na(TSS_grp)) %>% 
  dplyr::mutate(model = "ft")

### Pretrained
dnabert_pt <- pd$read_pickle('files/raw/DNABERT_custom_tata_scores_dictionary_attention_pretrained_annotations.pkl') %>%
  map(bind_cols) %>%
  list_rbind(names_to = "sequence") %>%
  dplyr::left_join(tata_sequences, by = "sequence") %>%
  dplyr::select(name, kmers, position, tag, TSS, repeat_Simple_repeat, starts_with('layer')) %>% 
  dplyr::mutate(across(where(~ is.numeric(.x)), ~ map_dbl(.x, ~ .x))) %>% 
  tidyr::pivot_longer(cols = starts_with('layer'),
                      names_to = 'id',
                      values_to = 'attention_scores') %>%
  tidyr::separate_wider_delim(cols = id, delim = "-", names = c("layer", "head"))

dnabert_pt_tss <- get_tss_windows(dnabert_pt)

dnabert_pt_tss_tb <- dnabert_pt %>% 
  dplyr::left_join(dnabert_pt_tss,
                   by = c("name", "kmers", "position"),
                   relationship = "many-to-many") %>% 
  dplyr::filter(!is.na(TSS_grp)) %>% 
  dplyr::mutate(model = "pt")

### Random init
dnabert_ri <- pd$read_pickle('files/raw/DNABERT_custom_tata_scores_dictionary_attention_random_init_annotations.pkl') %>%
  map(bind_cols) %>%
  list_rbind(names_to = "sequence") %>%
  dplyr::left_join(tata_sequences, by = "sequence") %>%
  dplyr::select(name, kmers, position, tag, TSS, repeat_Simple_repeat, starts_with('layer')) %>% 
  dplyr::mutate(across(where(~ is.numeric(.x)), ~ map_dbl(.x, ~ .x))) %>% 
  tidyr::pivot_longer(cols = starts_with('layer'),
                      names_to = 'id',
                      values_to = 'attention_scores') %>%
  tidyr::separate_wider_delim(cols = id, delim = "-", names = c("layer", "head"))

dnabert_ri_tss <- get_tss_windows(dnabert_ri)

dnabert_ri_tss_tb <- dnabert_ri %>% 
  dplyr::left_join(dnabert_ri_tss,
                   by = c("name", "kmers", "position"),
                   relationship = "many-to-many") %>% 
  dplyr::filter(!is.na(TSS_grp)) %>% 
  dplyr::mutate(model = "ri")

### Plot
gg_zscoreC <- bind_rows(dnabert_ft_tss_tb, dnabert_pt_tss_tb, dnabert_ri_tss_tb)  %>% 
  dplyr::filter(layer == "layer6" & head == "head9" & tag == 1) %>% 
  dplyr::group_by(model, TSS_position) %>% 
  dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop",
                   TSS = mean(TSS)) %>% 
  dplyr::mutate(model = factor(model,
                               levels = c("pt", "ft", "ri"),
                               labels = c("Pre-trained", "Fine-tuned", "Random initialized"))) %>% 
  ggplot(aes(x = TSS_position)) +
  geom_line(aes(y = attention_scores, color = model), linewidth = 1) +
  labs(subtitle = "layer6-head9",
       x = "TSS position (bp)",
       y = "Mean attention\nscore",
       color = "") +
  scale_color_manual(values = c("Pre-trained" = "#e07a5f", "Fine-tuned" = "#81b29a", "Random initialized" = "#5B6386")) +
  theme_bw() +
  theme(plot.subtitle = element_text(size = 13, hjust = 0.5),
        axis.title = element_text(size = 13),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 13),
        legend.text = element_text(size = 11),
        legend.position = "bottom")

## D) - PT vs RI scGPT ----
gg_zscoreD <- scgpt_zscore_files %>% 
  dplyr::filter(training %in% c("pt", "ri")) %>% 
  tidyr::pivot_wider(names_from = training,
                     values_from = zscore) %>% 
  dplyr::mutate(distance_diag = abs(pt - ri)/sqrt(2),
                sequence = factor(sequence,
                                  levels = c(0, 1),
                                  labels = c("Oligodendrocyte A", "Astrocyte")),
                label = glue::glue("{layer}-{feature}"),
                feature = ifelse(feature == "expression",
                                 feature,
                                 "other"),
                feature = factor(feature,
                                 levels = c("expression", "other"),
                                 labels = c("Expression", "Other"))) %>% 
  dplyr::group_by(sequence) %>% 
  dplyr::mutate(selected = ifelse(distance_diag > (mean(distance_diag)+sd(distance_diag*3)), "selected", "no_selected")) %>% 
  dplyr::ungroup() %>% 
  ggplot(aes(x = ri, y = pt)) +
  facet_wrap(~sequence) +
  geom_point(data = . %>% dplyr::filter(selected == "no_selected"), alpha = 0.7) +
  geom_point(data = . %>% dplyr::filter(selected == "selected"),
             aes(color = feature)) +
  geom_hline(aes(yintercept = 0), linetype = "dashed") +
  geom_vline(aes(xintercept = 0), linetype = "dashed") +
  geom_abline() +
  ggrepel::geom_label_repel(data = . %>% 
                              dplyr::filter(sequence == "Oligodendrocyte A") %>% 
                              dplyr::slice_max(order_by = distance_diag, n = 3),
                            aes(label = label),
                            box.padding = 1,
                            alpha = 0.9,
                            size = 3,
                            force = 2,
                            nudge_x = 50,
                            seed = 1234) +
  ggrepel::geom_label_repel(data = . %>% 
                              dplyr::filter(sequence == "Astrocyte") %>% 
                              dplyr::slice_max(order_by = distance_diag, n = 3),
                            aes(label = label),
                            alpha = 0.9,
                            box.padding = 1,
                            size = 3,
                            force = 2,
                            nudge_x = 50,
                            seed = 1234) +
  xlim(c(-50, 210)) +
  ylim(c(-50, 210)) +
  scale_color_manual(values = c("Expression" = "#bc4749", "Other" = "#9f86c0")) +
  labs(x = "Random initialized Z-Scores",
       y = "Pre-trained Z-Scores",
       color = "Significantly changed features") +
  theme_minimal() +
  theme(plot.title = element_text(size = 15, hjust = 0.5),
        plot.subtitle = element_text(size = 13, hjust = 0.5),
        panel.border = element_rect(fill = NA),
        strip.text = element_text(size = 12),
        axis.ticks = element_line(),
        axis.title = element_text(size = 13),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 13),
        legend.text = element_text(size = 11),
        legend.position = "top")
## E) - PT vs FT scGPT ----
gg_zscoreE <- scgpt_zscore_files %>% 
  dplyr::filter(training %in% c("pt", "ft")) %>% 
  tidyr::pivot_wider(names_from = training,
                     values_from = zscore) %>% 
  dplyr::mutate(distance_diag = abs(pt - ft)/sqrt(2),
                sequence = factor(sequence,
                                  levels = c(0, 1),
                                  labels = c("Oligodendrocyte A", "Astrocyte")),
                label = glue::glue("{layer}-{feature}"),
                feature = ifelse(feature == "expression",
                                 feature,
                                 "other"),
                feature = factor(feature,
                                 levels = c("expression", "other"),
                                 labels = c("Expression", "Other"))) %>% 
  dplyr::group_by(sequence) %>% 
  dplyr::mutate(selected = ifelse(distance_diag > (mean(distance_diag)+sd(distance_diag*3)), "selected", "no_selected")) %>% 
  dplyr::ungroup() %>% 
  ggplot(aes(x = ft, y = pt)) +
  facet_wrap(~sequence) +
  geom_point(data = . %>% dplyr::filter(selected == "no_selected"), alpha = 0.7) +
  geom_point(data = . %>% dplyr::filter(selected == "selected"),
             aes(color = feature)) +
  geom_hline(aes(yintercept = 0), linetype = "dashed") +
  geom_vline(aes(xintercept = 0), linetype = "dashed") +
  geom_abline() +
  ggrepel::geom_label_repel(data = . %>% 
                              dplyr::filter(sequence == "Oligodendrocyte A") %>% 
                              dplyr::slice_max(order_by = distance_diag, n = 3),
                            aes(label = label),
                            box.padding = 1,
                            alpha = 0.9,
                            size = 3,
                            force = 2,
                            seed = 1234) +
  ggrepel::geom_label_repel(data = . %>% 
                              dplyr::filter(sequence == "Astrocyte") %>% 
                              dplyr::slice_max(order_by = distance_diag, n = 3) %>% 
                              dplyr::mutate(label = case_when(label == "layer0_head0-GOBP regulation of cell differentiation" ~ "layer0_head0-GOBP\ncell differentiation",
                                                              label == "layer0_head0-GOBP positive regulation of developmental process" ~ "layer0_head0-GOBP\ndevelopmental process",
                                                              label == "layer0_head0-GOCC envelope" ~ "layer0_head0-GOCC\nenvelope")),
                            aes(label = label),
                            box.padding = 1,
                            alpha = 0.9,
                            size = 3,
                            force = 3,
                            force_pull = 2,
                            seed = 1234) +
  xlim(c(-50, 210)) +
  ylim(c(-50, 210)) +
  scale_color_manual(values = c("Expression" = "#bc4749", "Other" = "#9f86c0")) +
  labs(x = "Fine-tuned Z-Scores",
       y = "Pre-trained Z-Scores",
       color = "Significantly changed features") +
  theme_minimal() +
  theme(plot.title = element_text(size = 15, hjust = 0.5),
        plot.subtitle = element_text(size = 13, hjust = 0.5),
        panel.border = element_rect(fill = NA),
        strip.text = element_text(size = 12),
        axis.ticks = element_line(),
        axis.title = element_text(size = 13),
        axis.text = element_text(size = 12),
        legend.position = "")

## F) - Envelope feature for scGPT ----
scgpt_envelope_df <- map(names(scgpt_df_list), \(x) {
  scgpt_df_list[[x]] %>% 
    dplyr::filter(layer == "layer0" & head == "head0" & label %in% c("oligodendrocyte A", "astrocyte")) %>% 
    dplyr::mutate(model = str_split_i(x, "_", 2))
}) %>% 
  bind_rows() %>% 
  dplyr::mutate(attention_scores = as.numeric(attention_scores)) %>% 
  dplyr::rename(GOCC_envelope = `GOCC envelope`)
gg_zscoreF <- scgpt_envelope_df %>% 
  dplyr::mutate(model = factor(model,
                               levels = c("pt", "ft", "ri"),
                               labels = c("Pre-trained", "Fine-tuned", "Random initialized")),
                GOCC_envelope = factor(GOCC_envelope,
                                       levels = c(0,1),
                                       labels = c("Genes related to GOCC envelope", "Genes not related to GOCC envelope")),
                label = factor(label,
                               levels = c("oligodendrocyte A", "astrocyte"),
                               labels = c("Oligodendrocyte A", "Astrocyte"))) %>% 
  ggplot(aes(x = GOCC_envelope, y = attention_scores, color = model)) +
  facet_grid(~label+GOCC_envelope, scales = "free_x") +
  geom_jitter(alpha = 0.7) +
  labs(subtitle = "layer0-head0",
       y = "Attention score",
       color = "") +
  scale_color_manual(values = c("Pre-trained" = "#e07a5f", "Fine-tuned" = "#81b29a", "Random initialized" = "#5B6386")) +
  theme_bw() +
  theme(strip.background = element_blank(),
        plot.subtitle = element_text(size = 13, hjust = 0.5),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 13),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 12),
        axis.ticks.x = element_blank(),
        legend.title = element_text(size = 13),
        legend.text = element_text(size = 11),
        legend.position = "bottom")

## Plot - Figure 3 ----
gg_zscoreAB <- ggarrange(gg_zscoreA, gg_zscoreB,
                         ncol = 2,
                         labels = c("A", "B"),
                         common.legend = TRUE,
                         legend = "top",
                         legend.grob = get_legend(gg_zscoreB))
gg_zscoreAB <- annotate_figure(gg_zscoreAB,
                               top = text_grob("DNABERT - TATA", size = 15))
gg_zscoreDE <- ggarrange(gg_zscoreD, gg_zscoreE,
                         ncol = 2,
                         labels = c("D", "E"),
                         common.legend = TRUE,
                         legend = "top",
                         legend.grob = get_legend(gg_zscoreD))
gg_zscoreDE <- annotate_figure(gg_zscoreDE,
                               top = text_grob("scGPT - Multiple Sclerosis", size = 15))
gg_zscore <- ggarrange(gg_zscoreAB, gg_zscoreC, gg_zscoreDE, gg_zscoreF,
                       ncol = 1,
                       heights = c(0.3,0.2,0.3,0.2),
                       labels = c("", "C", "", "F"))

ggsave('Figure3.png',
       plot = gg_zscore,
       width = 3000,
       height = 4200,
       units = "px",
       # dpi = 1000,
       bg = "white")


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #

# Ablation analysis - Figure 4 ----
## DNABERT - TATA ----
### Open files ----
dnabert_ablation_files <- list.files('attentions/DNABERT/ablations/ablation_results/TATA', pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$")
names(dnabert_ablation_files) <- dnabert_ablation_files
dnabert_ablation_df <- set_names(dnabert_ablation_files) %>% 
  map(\(x) {read_delim(glue::glue("attentions/DNABERT/ablations/ablation_results/TATA/{x}"), delim = "\t", show_col_types = FALSE)}) %>% 
  list_rbind(names_to = "experiment")

### Calculate ROC ----
dnabert_ablation_df_roc <- dnabert_ablation_df %>% 
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
  dplyr::ungroup()

### Model factor values ----
dnabert_ablation_summary <- read_delim('attentions/DNABERT/ablations/ablation_results/TATA/TSS_ablation_summary.tsv', delim = "\t", show_col_types = FALSE) %>% 
  dplyr::mutate(Model = str_replace(Model, " Ablated", ""),
                Model = str_replace(Model, "TSS ", "")) %>% 
  dplyr::pull(Model)
#### Color
color <- c("black", 
           "#99e2b4", "#78c6a3", "#56ab91", "#358f80", "#14746f", "#036666",
           "#e01e37", "#c71f37", "#bd1f36", "#a71e34", "#85182a", "#641220")
names(color) <- dnabert_ablation_summary
#### Linetype
linetype <- c(1, rep(1,6), rep(2,6))
names(linetype) <- dnabert_ablation_summary
#### Linewidth
linewidth <- c(1.5, rep(1,6), rep(1,6))
names(linewidth) <- dnabert_ablation_summary

### Plot ----
gg_4a <- dnabert_ablation_df_roc %>% 
  dplyr::mutate(model = factor(model,
                               levels = c("baseline",
                                          "important_5percent", "important_10percent", "important_20percent", "important_30percent", "important_40percent", "important_50percent",
                                          "unimportant_5percent", "unimportant_10percent", "unimportant_20percent", "unimportant_30percent", "unimportant_40percent", "unimportant_50percent"),
                               labels = dnabert_ablation_summary)) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "DNABERT - TATA",
       subtitle = "feature: TSS",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

## NT - TATA ----
### Open files ----
nt_ablation_files <- list.files('attentions/nucleotide-transformer/ablations/ablation_results/custom_TATA', pattern = ".(TSS)*(percent_results\\.tsv|baseline_results\\.tsv)$")
names(nt_ablation_files) <- nt_ablation_files
nt_ablation_df <- set_names(nt_ablation_files) %>% 
  map(\(x) {read_delim(glue::glue("attentions/nucleotide-transformer/ablations/ablation_results/custom_TATA/{x}"), delim = "\t", show_col_types = FALSE)}) %>% 
  list_rbind(names_to = "experiment")

### Calculate ROC ----
nt_ablation_df_roc <- nt_ablation_df %>% 
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
  dplyr::ungroup()

### Model factor values ----
nt_ablation_summary <- read_delim('attentions/nucleotide-transformer/ablations/ablation_results/custom_TATA/TSS_ablation_summary.tsv', delim = "\t", show_col_types = FALSE) %>% 
  dplyr::mutate(Model = str_replace(Model, " Ablated", ""),
                Model = str_replace(Model, "TSS ", "")) %>% 
  dplyr::pull(Model)
#### Color
color <- c("black", 
           "#99e2b4", "#78c6a3", "#56ab91", "#358f80", "#14746f", "#036666",
           "#e01e37", "#c71f37", "#bd1f36", "#a71e34", "#85182a", "#641220")
names(color) <- nt_ablation_summary
#### Linetype
linetype <- c(1, rep(1,6), rep(2,6))
names(linetype) <- nt_ablation_summary
#### Linewidth
linewidth <- c(1.5, rep(1,6), rep(1,6))
names(linewidth) <- nt_ablation_summary

### Plot ----
gg_4b <- nt_ablation_df_roc %>% 
  dplyr::mutate(model = factor(model,
                               levels = c("baseline",
                                          "important_5percent", "important_10percent", "important_20percent", "important_30percent", "important_40percent", "important_50percent",
                                          "unimportant_5percent", "unimportant_10percent", "unimportant_20percent", "unimportant_30percent", "unimportant_40percent", "unimportant_50percent"),
                               labels = nt_ablation_summary)) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "Nucleotide Transformer - TATA",
       subtitle = "feature: TSS",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())


## scGPT - MS ----
### Open files ----
scgpt_ablation_files <- list.files('attentions/scGPT/ablations/ablation_results/scgpt_ms/GOCC neuron projection', pattern = ".*(percent_results\\.tsv|baseline_results\\.tsv)$")
names(dnabert_ablation_files) <- scgpt_ablation_files
scgpt_ablation_df <- set_names(scgpt_ablation_files) %>% 
  map(\(x) {read_delim(glue::glue("attentions/scGPT/ablations/ablation_results/scgpt_ms/GOCC neuron projection/{x}"), delim = "\t", show_col_types = FALSE)}) %>% 
  list_rbind(names_to = "experiment")

### Calculate ROC ----
scgpt_ablation_df_roc <- scgpt_ablation_df %>% 
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
                              names = paste0("x",0:17)) %>% 
  dplyr::mutate(across(starts_with("x"), ~ as.numeric(.x))) %>% 
  dplyr::group_by(model)
scgpt_ablation_df_roc <- scgpt_ablation_df_roc %>% 
  yardstick::roc_curve(truth = true_label,
                       any_of(unique(scgpt_ablation_df_roc$true_label))) %>% 
  dplyr::group_by(model, .level) %>% 
  dplyr::mutate(nrow = row_number()) %>% 
  dplyr::group_by(model, nrow) %>% 
  dplyr::summarise(specificity = mean(specificity),
                   sensitivity = mean(sensitivity)) %>% 
  dplyr::ungroup()

### Model factor values ----
scgpt_ablation_summary <- read_delim('attentions/scGPT/ablations/ablation_results/scgpt_ms/GOCC neuron projection/ms_ablation_summary.tsv', delim = "\t", show_col_types = FALSE) %>% 
  dplyr::mutate(Model = str_replace(Model, " Ablated", ""),
                Model = str_replace(Model, "TSS ", "")) %>% 
  dplyr::pull(Model)
#### Color
color <- c("black", 
           "#99e2b4", "#78c6a3", "#56ab91", "#358f80", "#14746f", "#036666",
           "#e01e37", "#c71f37", "#bd1f36", "#a71e34", "#85182a", "#641220")
names(color) <- scgpt_ablation_summary
#### Linetype
linetype <- c(1, rep(1,6), rep(2,6))
names(linetype) <- scgpt_ablation_summary
#### Linewidth
linewidth <- c(1.5, rep(1,6), rep(1,6))
names(linewidth) <- scgpt_ablation_summary

### Plot ----
gg_4c <- scgpt_ablation_df_roc %>% 
  dplyr::mutate(model = factor(model,
                               levels = c("baseline",
                                          "important_5percent", "important_10percent", "important_20percent", "important_30percent", "important_40percent", "important_50percent",
                                          "unimportant_5percent", "unimportant_10percent", "unimportant_20percent", "unimportant_30percent", "unimportant_40percent", "unimportant_50percent"),
                               labels = scgpt_ablation_summary)) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model, linewidth = model)) +
  geom_abline(linetype = 3, linewidth = 1.5) +
  geom_path() +
  scale_color_manual(values = color) +
  scale_linetype_manual(values = linetype) +
  scale_linewidth_manual(values = linewidth) +
  labs(title = "scGPT - Multiple Sclerosis",
       subtitle = "feature: GOCC neuron projection",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.title = element_blank())

## Plot - Figure 4 ----
ggpubr::ggarrange(gg_4a, gg_4b, gg_4c,
                  nrow = 2,
                  ncol = 2,
                  labels = c("A", "B", "C"),
                  common.legend = TRUE,
                  legend = "right")
