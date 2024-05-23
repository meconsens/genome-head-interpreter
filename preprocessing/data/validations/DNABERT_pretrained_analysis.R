library(qs)
library(reticulate)
library(readr)
library(plyr)
library(dplyr)
library(tidyr)
library(stringr)
library(rjson)
library(ggplot2)
library(ggpubr)
library(ggnewscale)
library(viridis)
library(ggseqlogo)
library(ggsci)
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


# DNABERT scores ----
## Preprocessing ----
# scores <- pd$read_pickle('/DNABERT/enhancer_scores_dictionary_attention_pretrained_updated_annotations.pkl')
# scores <- ldply(scores, bind_cols) %>%
#   as_tibble() %>%
#   dplyr::rename(id = `.id`) %>%
#   tidyr::pivot_longer(cols = starts_with('layer'), names_to = 'layer', values_to = 'attention_scores') %>%
#   dplyr::mutate(head = factor(str_split_i(layer, '-', 2),
#                               levels = paste0('head', seq(0,11))),
#                 layer = factor(str_split_i(layer, '-', 1),
#                                levels = paste0('layer', seq(0,11))))
# id_names_map <- scores %>%
#   dplyr::select(id) %>%
#   dplyr::distinct() %>%
#   dplyr::mutate(name = paste0("sequence", seq(1,nrow(.))))
# scores <- scores %>%
#   dplyr::left_join(id_names_map, by = "id") %>%
#   dplyr::select(id, name, kmers, starts_with('position'), layer, head, attention_scores, everything())
# qsave(scores, file = "/DNABERT/enhancer_scores_dictionary_attention_pretrained_updated_annotations_preprocessed.qs")

## Open scores ----
scores <- qread("/DNABERT/enhancer_scores_dictionary_attention_pretrained_updated_annotations_preprocessed.qs")


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# General plots ----
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
       subtitle = "DNABERT",
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
       subtitle = "DNABERT",
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
                     ymax = ifelse(ymax > 1, 1, ymax),
                     n = n()) %>% 
    dplyr::group_by(head) %>% 
    dplyr::mutate(n = n/max(n)) %>% 
    ggplot(aes(x = position)) +
    facet_wrap(~head) +
    geom_line(aes( y = attention_scores, group = head, color = attention_scores),
              linewidth = 0.8) +
    geom_ribbon(aes(ymin = ymin, ymax = ymax), 
                color = 'black', alpha = 0,
                linewidth = 0.1, linetype = 'dashed') +
    geom_line(aes(y = n), color = '#FAB061', linewidth = 0.2) +
    scale_color_viridis_c(option = "G", direction = -1) +
    labs(title = glue::glue('Mean attention for {layer}'),
         subtitle = "DNABERT",
         x = 'Nucleotide position',
         caption = "Ribbon shows two standard deviations") +
    scale_y_continuous("Attention score", limits = c(0,1), sec.axis = dup_axis(name = "Sequences (%)")) +
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
  ggsave(filename = glue::glue('/DNABERT/plots/attention_vs_position/{layer}.png'), plot = gg, bg = 'white')
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
coefficients <- ldply(fromJSON(file = "/DNABERT/coefficients/DNABERT_pretrained.json"), extract_coefficients) %>% 
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
# write_csv(filter_coefficients, file = '/DNABERT/coefficients/dnabert_filter_coefficients_pretrained.csv')
# id_correlations <- adply(filter_coefficients, 1, get_id_correlations, .progress = 'time') %>%
#   as_tibble() %>%
#   qsave(., file = '/DNABERT/coefficients/dnabert_individual_coefficients_pretrained.qs')
# write_csv(id_correlations, file = '/DNABERT/coefficients/dnabert_individual_coefficients_pretrained.csv')
id_names_map <- scores %>%
  dplyr::select(id,name) %>%
  dplyr::distinct()
id_correlations <- qread('/DNABERT/coefficients/dnabert_individual_coefficients_pretrained.qs') %>% 
  dplyr::filter(!is.na(cor) & abs(cor) > 0.2) %>% 
  dplyr::left_join(id_names_map, by = 'id')


## Plots ----
plot_validation <- function(df, name.f){
  df %>% 
    ggplot(aes(x = position, y = attention_scores, color = color, group = group)) +
    facet_wrap(~name, ncol = 1, scales = 'free_y') +
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
    scale_fill_manual(values = c('white', '#067BC2', '#e59500', '#bb3e03'), name = "") +
    scale_color_manual(values = c('#d2d4c8', '#067BC2', '#e59500', '#bb3e03', 'black'), name = "") +
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
### layer3-head5 + bZIP, p53 ----
quantile(scores$`Basic leucine zipper factors (bZIP)`[which(scores$`Basic leucine zipper factors (bZIP)` != 0)])
quantile(scores$`p53 domain factors`[which(scores$`p53 domain factors` != 0)])
  
scores %>% 
  dplyr::filter(name %in% c('sequence222', 'sequence1490', 'sequence1194', 'sequence922')) %>% 
  dplyr::filter(layer == 'layer3' & head == 'head5') %>% 
  dplyr::select(name, position, attention_scores, `Basic leucine zipper factors (bZIP)`, `p53 domain factors`) %>% 
  dplyr::group_by(name) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(`Basic leucine zipper factors (bZIP)` > 3 & `p53 domain factors` >= 1,
                               "Overlap",
                               ifelse(`Basic leucine zipper factors (bZIP)` > 3,
                                      "Basic leucine zipper factors (bZIP)",
                                      ifelse(`p53 domain factors` >= 1,
                                             'p53 domain factors',
                                             "No Feature"))),
                color = factor(color, levels = c("No Feature", "Basic leucine zipper factors (bZIP)", 'p53 domain factors', 'Overlap')),
                position2 = position+1,
                group = "group",
                name = factor(name,
                              levels = c('sequence222', 'sequence922', 'sequence1194', 'sequence1490'),
                              labels = c('Sequence 222', 'Sequence 922', 'Sequence 1194', 'Sequence 1490'))) %>% 
  plot_validation(., name.f = "layer3-head5")

### layer0-head8 + GC, bHLH ----
quantile(scores$GC)
quantile(scores$`Basic helix-loop-helix factors (bHLH)`[which(scores$`Basic helix-loop-helix factors (bHLH)` != 0)])

scores %>% 
  dplyr::filter(name %in% c('sequence59', 'sequence1599', 'sequence517', 'sequence977')) %>% 
  dplyr::filter(layer == 'layer0' & head == 'head8') %>% 
  dplyr::select(name, position, attention_scores, GC, `Basic helix-loop-helix factors (bHLH)`) %>% 
  dplyr::group_by(name) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(GC > 0.6 & `Basic helix-loop-helix factors (bHLH)` > 3,
                               "Overlap",
                               ifelse(GC > 0.6,
                                      "High GC",
                                      ifelse(`Basic helix-loop-helix factors (bHLH)` > 3,
                                             'Basic helix-loop-helix factors (bHLH)',
                                             "No Feature"))),
                color = factor(color, levels = c("No Feature", "High GC", 'Basic helix-loop-helix factors (bHLH)', 'Overlap')),
                position2 = position+1,
                group = "group",
                name = factor(name, 
                              levels = c('sequence59', 'sequence517', 'sequence977', 'sequence1599'),
                              labels = c('Sequence 59', 'Sequence 517', 'Sequence 977', 'Sequence 1599'))) %>% 
  plot_validation(., name.f = "layer0-head8")

### layer5-head11 - TSS + TATA ----
quantile(scores$TSS[which(scores$TSS != 0)])
quantile(scores$`TATA-binding proteins`[which(scores$`TATA-binding proteins` != 0)])

scores %>% 
  dplyr::filter(name %in% c('sequence1029', 'sequence1239', 'sequence1700', 'sequence1794')) %>% 
  dplyr::filter(layer == 'layer5' & head == 'head11') %>% 
  dplyr::select(name, position, attention_scores, TSS, `TATA-binding proteins`) %>% 
  dplyr::group_by(name) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(TSS >= 1 & `TATA-binding proteins` >= 1,
                               "Overlap",
                               ifelse(TSS >= 1,
                                      "TSS",
                                      ifelse(`TATA-binding proteins` >= 1,
                                             'TATA-binding proteins',
                                             "No Feature"))),
                color = factor(color, levels = c("No Feature", "TSS", "TATA-binding proteins", 'Overlap')),
                position2 = position+1,
                group = "group",
                name = factor(name,
                              levels = c('sequence1029', 'sequence1239', 'sequence1700', 'sequence1794'),
                              labels = c('Sequence 1029', 'Sequence 1239', 'Sequence 1700', 'Sequence 1794'))) %>% 
  plot_validation(., name.f = "layer5-head11")

### layer1-head11 - SINE + DNA ----
quantile(scores$repeat_SINE[which(scores$repeat_SINE != 0)])
quantile(scores$repeat_DNA[which(scores$repeat_DNA != 0)])

scores %>% 
  dplyr::filter(name %in% c('sequence185', 'sequence824', 'sequence891', 'sequence971')) %>% 
  dplyr::filter(layer == 'layer1' & head == 'head11') %>% 
  dplyr::select(name, position, attention_scores, repeat_SINE, repeat_DNA) %>% 
  dplyr::group_by(name) %>% 
  dplyr::mutate(ymin = mean(attention_scores)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(color = ifelse(repeat_SINE >= 1 & repeat_DNA >= 1,
                               "Overlap",
                               ifelse(repeat_SINE >= 1,
                                      "Repeat SINE",
                                      ifelse(repeat_DNA >= 1,
                                             'Repeat DNA',
                                             "No Feature"))),
                color = factor(color, levels = c("No Feature", "Repeat SINE", "Repeat DNA", 'Overlap')),
                position2 = position+1,
                group = "group",
                name = factor(name, 
                              levels = c('sequence185', 'sequence824', 'sequence891', 'sequence971'),
                              labels = c('Sequence 185', 'Sequence 824', 'Sequence 891', 'Sequence 971'))) %>% 
  plot_validation(., name.f = "layer1-head11")


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# kmer analysis ----
## Most activated kmers ----
# most_activated_kmers <- scores %>% 
#   dplyr::group_by(layer, head) %>% 
#   dplyr::slice_max(attention_scores, n=1000) %>% 
#   dplyr::select(name, kmers, position, layer, head, attention_scores) %>% 
#   dplyr::ungroup()
# write_csv(most_activated_kmers, file = "/DNABERT/kmer_analysis/most_activated_kmers_pretrained.csv")
most_activated_kmers <- read_csv("/DNABERT/kmer_analysis/most_activated_kmers_pretrained.csv")

## kmer abundance ----
# most_activated_kmers_analysis <- most_activated_kmers %>% 
#   dplyr::group_by(layer, head, kmers) %>% 
#   dplyr::summarise(n = n()) %>% 
#   dplyr::group_by(layer, head) %>% 
#   dplyr::mutate(perc = n/sum(n)*100) %>% 
#   dplyr::ungroup()
# write_csv(most_activated_kmers_analysis, file = "/DNABERT/kmer_analysis/most_activated_kmers_analysis_pretrained.csv")
most_activated_kmers_analysis <- read_csv("/DNABERT/kmer_analysis/most_activated_kmers_analysis_pretrained.csv")

### Barplot by layer ----
most_activated_kmers_analysis %>% 
  dplyr::group_by(layer, kmers) %>% 
  dplyr::summarise(n = sum(n)) %>% 
  dplyr::group_by(layer) %>% 
  dplyr::slice_max(n, n=10) %>% 
  dplyr::ungroup() %>% 
  dplyr::select(layer, kmers, n) %>% 
  dplyr::mutate(kmers = factor(kmers, levels = unique(kmers)),
                layer = factor(layer,
                               levels = paste0("layer", 0:11))) %>%
  ggplot(aes(x = kmers, y = n, fill = n)) +
  facet_wrap(~layer, scales = "free_x") +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_viridis_c(option = "F", direction = -1, end = 0.8) +
  labs(title = "Barplot for the most abundant kmers by layer",
       subtitle = "DNABERT pretrained",
       x = "kmers",
       y = "Abundance") +
  ylim(c(0, 500)) +
  theme_minimal() +
  theme(panel.border = element_rect(fill = NA),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        axis.ticks = element_line(),
        axis.title = element_text(size = 13),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 9),
        legend.position = "none")
### Heatmap by head ----
most_activated_kmers_analysis %>% 
  dplyr::group_by(layer, head) %>% 
  dplyr::summarise(n = max(n)) %>% 
  dplyr::ungroup() %>% 
  tidyr::pivot_wider(names_from = head, values_from = n, values_fill = 0) %>% 
  tidyr::pivot_longer(cols = starts_with('head'), names_to = "head", values_to = "n") %>% 
  dplyr::mutate(layer = factor(layer,
                               levels = paste0("layer", 11:0)),
                head = factor(head,
                              levels = paste0("head", 0:11))) %>% 
  ggplot(.,
         aes(x = head, y = layer, fill = n)) +
  geom_tile() +
  scale_fill_viridis_c(option = "F", direction = -1,
                       begin = 0.2,
                       limits = c(0,220)) +
  labs(title = "Heatmap for the most abundant kmer by head",
       subtitle = "DNABERT pretrained",
       fill = "Abundance") +
  theme_minimal() +
  theme(axis.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text = element_text(size = 10),
        legend.position = "right")

## Logo ----
most_activated_kmers %>% 
  tidyr::separate(kmers, sep = "", into = paste0('nt',0:6)) %>% 
  dplyr::select(-nt0, -name, -position, -attention_scores) %>% 
  dplyr::mutate(layer = factor(layer,
                               levels = paste0("layer", 0:11)),
                head = factor(head,
                              levels = paste0("head", 0:11))) %>% 
  dplyr::group_by(layer) %>% 
  dplyr::group_split() %>% 
  l_ply(., function(y){
    layern <- y$layer[1]
    headn <- y$head[1]
    gg_list <- y %>%
      dplyr::group_by(head) %>% 
      dplyr::group_split() %>% 
      llply(., function(x){
        layern <- x$layer[1]
        headn <- x$head[1]
        tmp <- x %>% 
          dplyr::select(starts_with('nt')) %>% 
          tidyr::pivot_longer(cols = everything(),
                              names_to = "position",
                              values_to = "nucleotides") %>% 
          dplyr::group_by(position, nucleotides) %>% 
          dplyr::summarise(n = n()) %>% 
          dplyr::ungroup() %>% 
          tidyr::pivot_wider(names_from = position,
                             values_from = n) %>% 
          dplyr::select(-nucleotides) %>% 
          as.matrix()
        rownames(tmp) <- c('A', 'C', 'G', 'T')
        cs <- make_col_scheme(chars = c('A', 'C', 'G', 'T'),
                              groups = c('gr1', 'gr2', 'gr3', 'gr4'), 
                              cols = pal_jama()(4))
        gglogo <- ggseqlogo(tmp,
                            method = "prob",
                            facet = "grid",
                            col_scheme = cs) +
          geom_vline(aes(xintercept = -Inf)) +
          geom_hline(aes(yintercept = -Inf)) +
          labs(title = glue::glue("{layern}_{headn}")) +
          theme(legend.position = "none",
                plot.title = element_text(size = 9),
                axis.title = element_text(size = 8),
                axis.text = element_text(size = 7))
        return(gglogo)
      })
    gg_final <- ggarrange(plotlist = gg_list)
    ggsave(filename = glue::glue('/DNABERT/plots_pretrained/kmer_analysis/logos/logo_{layern}_pretrained.png'), plot = gg_final, bg = 'white')
  })

