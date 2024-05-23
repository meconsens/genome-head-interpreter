library(qs)
library(reticulate)
library(readr)
library(plyr)
library(dplyr)
library(tidyr)
library(stringr)
library(rjson)
library(ggplot2)
library(ggnewscale)
library(viridis)
pd <- import("pandas")


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Options ----
options(dplyr.summarise.inform = FALSE)
layer_factors <- c(expand.grid(paste0('head', 0:11), paste0('layer', 0:11))) %>% 
  as_tibble() %>% 
  dplyr::mutate(layer = paste(Var2, Var1, sep='-')) %>% 
  dplyr::pull(layer)


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Enformer scores ----
## Preprocessing ----
# scores <- pd$read_pickle('/Users/ander/Laboratorios/Laboratorio_Lincoln/projects/genomic_interpretability/Enformer/enformer_scores_dictionary_attention_annotations.pkl')
# scores <- ldply(scores, bind_cols) %>%
#   as_tibble() %>% 
#   dplyr::rename(id = `.id`) %>% 
#   tidyr::pivot_longer(cols = starts_with('layer'), names_to = 'layer', values_to = 'attention_scores') %>% 
#   dplyr::mutate(head = factor(str_split_i(layer, '-', 2),
#                               levels = paste0('head', seq(0,7))),
#                 layer = factor(str_split_i(layer, '-', 1),
#                                levels = paste0('layer', seq(0,10)))) %>% 
#   dplyr::select(id, starts_with('position'), layer, head, attention_scores, everything())
# qsave(scores, file = "/Users/ander/Laboratorios/Laboratorio_Lincoln/projects/genomic_interpretability/Enformer/enformer_scores_dictionary_attention_annotations_preprocessed.qs")

## Open scores ----
scores <- qread("/Users/ander/Laboratorios/Laboratorio_Lincoln/projects/genomic_interpretability/Enformer/enformer_scores_dictionary_attention_annotations_preprocessed.qs")

### Mean attention scores - No normalized
mean_scores <- scores %>% 
  dplyr::group_by(layer, head) %>% 
  dplyr::summarise(attention_scores = mean(attention_scores)) %>% 
  dplyr::ungroup()
mean_scores_layers <- mean_scores %>% 
  dplyr::group_by(layer) %>% 
  dplyr::summarise(attention_scores = mean(attention_scores)) %>% 
  dplyr::ungroup()
mean_scores_heads <- mean_scores %>% 
  dplyr::group_by(head) %>% 
  dplyr::summarise(attention_scores = mean(attention_scores)) %>% 
  dplyr::ungroup()


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Plots ----
## Heatmap ----
scores %>% 
  dplyr::select(layer, head, attention_scores) %>% 
  dplyr::group_by(layer, head) %>% 
  dplyr::summarise(attention_scores = mean(attention_scores)) %>% 
  dplyr::group_by(layer) %>% 
  dplyr::mutate(attention_scores = (attention_scores - min(attention_scores)) / (max(attention_scores) - min(attention_scores))) %>% 
  dplyr::ungroup() %>% 
  ggplot(.,
       aes(x = head, y = layer, fill = attention_scores)) +
  geom_tile() +
  scale_fill_viridis_c(option = 'G', direction = -1, limits = c(0,1)) +
  labs(title = "Heatmap for mean attention score by layer-head",
       subtitle = "Enformer",
       fill = "Attention score") +
  theme_minimal() +
  theme(axis.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 10),
        legend.position = "right")

## Per sequence score ----
scores %>% 
  dplyr::select(id, layer, head, attention_scores) %>% 
  dplyr::group_by(id, layer, head) %>% 
  dplyr::summarise(attention_scores = mean(attention_scores)) %>% 
  dplyr::group_by(id, layer) %>% 
  dplyr::mutate(attention_scores = (attention_scores - min(attention_scores)) / (max(attention_scores) - min(attention_scores))) %>% 
  dplyr::group_by(layer, head) %>% 
  dplyr::mutate(color = mean(attention_scores)) %>% 
  ggplot(.,
         aes(x = head, y = attention_scores, fill = color)) +
  facet_wrap(~layer) +
  geom_boxplot(alpha = 0.7,
               outlier.alpha = 0.5,
               outlier.size = 1) +
  scale_fill_viridis_c(option = 'C', direction = -1) +
  labs(title = "Per sequence score",
       subtitle = "Enformer",
       x = "",
       y = "Attention score") +
  theme_minimal() +
  theme(panel.border = element_rect(fill = NA),
        panel.grid.minor.x = element_blank(),
        axis.ticks = element_line(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 10),
        legend.position = "none")

## Mean attention vs. position ----
l_ply(scores %>% dplyr::select(layer, head, position, attention_scores) %>% dplyr::group_split(layer), function(x){
  layer <- as.character(x$layer[1])
  gg <- x %>%
    dplyr::group_by(head, position) %>%
    dplyr::summarise(sd = sd(attention_scores),
                     attention_scores = mean(attention_scores),
                     ymin = attention_scores-sd*2,
                     ymax = attention_scores+sd*2,
                     ymin = ifelse(ymin < 0, 0, ymin),
                     ymax = ifelse(ymax > 1, 1, ymax)) %>% 
    ggplot(aes(x = position)) +
    facet_wrap(~head) +
    geom_line(aes( y = attention_scores, group = head, color = attention_scores),
              linewidth = 0.8) +
    geom_ribbon(aes(ymin = ymin, ymax = ymax), 
                color = 'black', alpha = 0,
                linewidth = 0.1, linetype = 'dashed') +
    scale_color_viridis_c(option = "G", direction = -1) +
    labs(title = glue::glue('Mean attention for {layer}'),
         subtitle = "Enformer",
         y = 'Attention score',
         x = 'Nucleotide position',
         caption = "Ribbon shows two standard deviations") +
    ylim(c(0,1)) +
    theme_minimal() +
    theme(panel.border = element_rect(fill = NA),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          axis.ticks = element_line(),
          axis.title = element_text(size = 14),
          axis.title.x = element_blank(),
          axis.text.y = element_text(size = 10),
          axis.text.x = element_text(size = 10),
          legend.position = "none")
  ggsave(filename = glue::glue('/Users/ander/Laboratorios/Laboratorio_Lincoln/projects/genomic_interpretability/Enformer/plots/attention_vs_position/{layer}.png'), plot = gg, bg = 'white')
})


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Validations ----
extract_coefficients <- function(json.feature){
  sentences <- unlist(json.feature$sentences)

  features <- character()
  values <- numeric()
  for (i in seq_along(sentences)) {
    # If not numeric, it's a feature
    if (is.na(as.numeric(sentences[i]))) {
      if (length(features) > length(values)){
        values <- c(values, 0)
        features <- c(features, sentences[i])
      } else {
        features <- c(features, sentences[i])
      }
      # If numeric, it's a value
    } else {
      values <- c(values, as.numeric(sentences[i]))
    }
  }
  # In case the last feature is empty
  if (length(features) > length(values)){
    values <- c(values, 0)
  }

  tb <- data.frame(features, values)
  
  return(tb)
}
get_id_correlations <- function(x){
  df <- scores %>% 
    dplyr::filter(layer == x$layer & head == x$head) %>% 
    dplyr::select(attention_scores, id, {x$features}) %>%
    dplyr::rename(features = {x$features}) %>% 
    dplyr::group_by(id) %>% 
    dplyr::group_split() %>% 
    ldply(., function(y){
      df <- data.frame(id = unique(y$id),
                       cor = cor(y$attention_scores, y$features, method = "spearman"))
      return(df)
    })
  return(df)
}
coefficients <- ldply(fromJSON(file = "/Users/ander/Laboratorios/Laboratorio_Lincoln/projects/genomic_interpretability/Enformer/coefficients/enformer.json"), extract_coefficients) %>% 
  as_tibble() %>% 
  bind_rows() %>% 
  dplyr::rename(layer_head = `.id`) %>% 
  dplyr::mutate(layer = str_split_i(layer_head, "-", 1),
                head = str_split_i(layer_head, "-", 2))
filter_coefficients <- coefficients %>% 
  dplyr::filter(abs(values) > 4 & str_starts(features, "position", negate = TRUE) & str_starts(features, "Position", negate = TRUE)) %>% 
  dplyr::group_by(layer_head) %>% 
  dplyr::mutate(n = n()) %>% 
  dplyr::ungroup()
# write_csv(filter_coefficients, file = '/Users/ander/Laboratorios/Laboratorio_Lincoln/projects/genomic_interpretability/Enformer/coefficients/enformer_filter_coefficients.csv')
# id_correlations <- adply(filter_coefficients, 1, get_id_correlations, .progress = 'time') %>% 
#   as_tibble() %>% 
#   qsave(., file = '/Users/ander/Laboratorios/Laboratorio_Lincoln/projects/genomic_interpretability/Enformer/coefficients/enformer_individual_coefficients.qs')
id_correlations <- qread('/Users/ander/Laboratorios/Laboratorio_Lincoln/projects/genomic_interpretability/Enformer/coefficients/enformer_individual_coefficients.qs') %>% 
  dplyr::filter(!is.na(cor) & abs(cor) > 0.05)
  
## Plots ----
plot_validation <- function(df, name.f){
  df %>% 
    ggplot(aes(x = position, y = attention_scores, color = color, group = group)) +
    facet_wrap(~id, ncol = 1, scales = 'free_y') +
    geom_rect(data = . %>% dplyr::filter(attention_scores > ymin),
              aes(xmin = position, xmax = position2,
                  ymin = ymin, ymax = Inf,
                  fill = color),
              color = NA,
              alpha = 0.25) +
    geom_rect(data = . %>% dplyr::filter(attention_scores <= ymin),
              aes(xmin = position, xmax = position2,
                  ymin = -Inf, ymax = ymin,
                  fill = color),
              color = NA,
              alpha = 0.25) +
    geom_line(aes(color = color)) +
    scale_fill_manual(values = c('white', '#067BC2', '#e59500', '#558564'), name = "") +
    scale_color_manual(values = c('#d2d4c8', '#067BC2', '#e59500', '#558564', 'black'), name = "") +
    new_scale_color() +
    geom_hline(aes(yintercept = ymin, color = "Mean Attention"),
               linewidth = 0.3,
               linetype = "dashed") +
    scale_color_manual(values = c('black'), name = "", guide = guide_legend(order = 1)) +
    labs(title = name.f,
         y = "Attention score") +
    theme_minimal() +
    theme(panel.border = element_rect(fill = NA),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          axis.ticks = element_line(),
          axis.title = element_text(size = 14),
          axis.title.x = element_blank(),
          axis.text.y = element_text(size = 10),
          axis.text.x = element_text(size = 10),
          legend.position = "top")
}
### layer1-head2 + SINE ----
quantile(scores$repeat_SINE[which(scores$repeat_SINE != 0)])

scores %>% 
  dplyr::filter(id %in% c('ENSG00000150093_ITGB1', 'ENSG00000177565_TBL1XR1', 'ENSG00000187109_NAP1L1')) %>% 
  dplyr::filter(layer == 'layer1' & head == 'head2') %>% 
  dplyr::select(id, position, attention_scores, repeat_SINE) %>% 
  dplyr::group_by(id) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(repeat_SINE < 0.4, "No Feature", "repeat_SINE"),
                color = factor(color, 
                               levels = c("No Feature", "repeat_SINE"),
                               labels = c("No Feature", "Repeat SINE")),
                position2 = position+128,
                group = "group") %>% 
  plot_validation(., name.f = "layer1-head2")

### layer2-head2 + CRISPR_screening, repeat_DNA----
quantile(scores$CRISPR_screening[which(scores$CRISPR_screening != 0)])
quantile(scores$repeat_DNA[which(scores$repeat_DNA != 0)])

scores %>% 
  dplyr::filter(id %in% c('ENSG00000184182_UBE2F', 'ENSG00000106392_C1GALT1', 'ENSG00000182512_GLRX5')) %>% 
  dplyr::filter(layer == 'layer2' & head == 'head2') %>% 
  dplyr::select(id, position, attention_scores, CRISPR_screening, repeat_DNA) %>% 
  dplyr::group_by(id) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(CRISPR_screening < 0.57, "No Feature", "CRISPR_screening"),
                color = ifelse(repeat_DNA < 0.34, color, "repeat_DNA"),
                color = factor(color, 
                               levels = c("No Feature", "CRISPR_screening", "repeat_DNA"),
                               labels = c("No Feature", "Enhancers (screening)", "Repeat DNA")),
                position2 = position+128,
                group = "group") %>% 
  plot_validation(., name.f = "layer2-head2")

### layer2-head5 + CTCF_bound ----
quantile(scores$CTCF_bound[which(scores$CTCF_bound != 0)])

scores %>% 
  dplyr::filter(id %in% c('ENSG00000159176_CSRP1', 'ENSG00000164040_PGRMC2', 'ENSG00000186716_BCR')) %>% 
  dplyr::filter(layer == 'layer2' & head == 'head5') %>% 
  dplyr::select(id, position, attention_scores, CTCF_bound) %>% 
  dplyr::group_by(id) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(CTCF_bound < 0.4, "No Feature", "CTCF_bound"),
                color = factor(color, 
                               levels = c("No Feature", "CTCF_bound"),
                               labels = c("No Feature", "CTCF bound")),
                position2 = position+128,
                group = "group") %>% 
  plot_validation(., name.f = "layer2-head5")

### layer3-head4 + promoter ----
quantile(scores$promoters[which(scores$promoters != 0)])

scores %>% 
  dplyr::filter(id %in% c('ENSG00000185551_NR2F2', 'ENSG00000073417_PDE8A', 'ENSG00000265681_RPL17')) %>% 
  dplyr::filter(layer == 'layer3' & head == 'head4') %>% 
  dplyr::select(id, position, attention_scores, promoters) %>% 
  dplyr::group_by(id) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(promoters < 0.49, "No Feature", "promoters"),
                color = factor(color,
                               levels = c("No Feature", "promoters"),
                               labels = c("No Feature", "Promoters")),
                position2 = position+128,
                group = "group") %>% 
  plot_validation(., name.f = "layer3-head4")

### layer6-head4 + repeat_DNA ----
quantile(scores$repeat_DNA[which(scores$repeat_DNA != 0)])

scores %>% 
  dplyr::filter(id %in% c('ENSG00000159176_CSRP1', 'ENSG00000164040_PGRMC2', 'ENSG00000186716_BCR')) %>% 
  dplyr::filter(layer == 'layer6' & head == 'head4') %>% 
  dplyr::select(id, position, attention_scores, repeat_DNA) %>% 
  dplyr::group_by(id) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(repeat_DNA < 0.34, "No Feature", "repeat_DNA"),
                color = factor(color, 
                               levels = c("No Feature", "repeat_DNA"),
                               labels = c("No Feature", "Repeat DNA")),
                position2 = position+128,
                group = "group") %>% 
  plot_validation(., name.f = "layer6-head4")

### layer8-head3 - enhancers ----
quantile(scores$enhancers[which(scores$enhancers != 0)])

scores %>% 
  dplyr::filter(id %in% c('ENSG00000079785_DDX1', 'ENSG00000196693_ZNF33B', 'ENSG00000130340_SNX9')) %>% 
  dplyr::filter(layer == 'layer8' & head == 'head3') %>% 
  dplyr::select(id, position, attention_scores, enhancers) %>% 
  dplyr::group_by(id) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(enhancers < 0.5, "No Feature", "enhancers"),
                color = factor(color, 
                               levels = c("No Feature", "enhancers"),
                               labels = c("No Feature", "Enhancers")),
                position2 = position+128,
                group = "group") %>% 
  plot_validation(., name.f = "layer8-head3")

### layer9-head6 + tss, promoter, crispr_perturbation ----
quantile(scores$TSS[which(scores$TSS != 0)])
quantile(scores$promoters[which(scores$promoters != 0)])
quantile(scores$CRISPR_perturbation[which(scores$CRISPR_perturbation != 0)])

scores %>% 
  dplyr::filter(id %in% c('ENSG00000135046_ANXA1', 'ENSG00000114315_HES1', 'ENSG00000108819_PPP1R9B', 'ENSG00000183337_BCOR')) %>% 
  dplyr::filter(layer == 'layer9' & head == 'head6') %>% 
  dplyr::select(id, position, attention_scores, TSS, promoters, CRISPR_perturbation) %>% 
  dplyr::group_by(id) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(CRISPR_perturbation < 0.79, "No Feature", "CRISPR_perturbation"),
                color = ifelse(promoters < 0.49, color, "promoters"),
                color = ifelse(TSS < 0.046, color, "TSS"),
                color = factor(color,
                               levels = c("No Feature", "TSS", "promoters", "CRISPR_perturbation"),
                               labels = c("No Feature", "TSS", "Promoters", "Enhancers (perturbation)")),
                position2 = position+128,
                group = "group") %>% 
  plot_validation(., name.f = "layer9-head6")

### layer9-head6 - repeat_LINE ----
quantile(scores$repeat_LINE[which(scores$repeat_LINE != 0)])

scores %>% 
  dplyr::filter(id %in% c('ENSG00000184182_UBE2F', 'ENSG00000185559_DLK1', 'ENSG00000164659_ELAPOR2')) %>% 
  dplyr::filter(layer == 'layer9' & head == 'head6') %>% 
  dplyr::select(id, position, attention_scores, repeat_LINE) %>% 
  dplyr::group_by(id) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(repeat_LINE < 0.51, "No Feature", "repeat_LINE"),
                color = factor(color,
                               levels = c("No Feature", "repeat_LINE"),
                               labels = c("No Feature", "Repeat LINE")),
                position2 = position+128,
                group = "group") %>% 
  plot_validation(., name.f = "layer9-head6")

### layer10-head7 + enhancers ----
quantile(scores$enhancers[which(scores$enhancers != 0)])

scores %>% 
  dplyr::filter(id %in% c('ENSG00000067082_KLF6', 'ENSG00000131389_SLC6A6', 'ENSG00000123080_CDKN2C')) %>% 
  dplyr::filter(layer == 'layer10' & head == 'head7') %>% 
  dplyr::select(id, position, attention_scores, enhancers) %>% 
  dplyr::group_by(id) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(enhancers < 0.5, "No Feature", "enhancers"),
                color = factor(color, 
                               levels = c("No Feature", "enhancers"),
                               labels = c("No Feature", "Enhancers")),
                position2 = position+128,
                group = "group") %>% 
  plot_validation(., name.f = "layer10-head7")
