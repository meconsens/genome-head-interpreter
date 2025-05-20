library(qs)
library(reticulate)
library(tidyverse)
library(Biostrings)
pd <- import("pandas")
options(dplyr.summarise.inform = FALSE)


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Sequences ----
## TATA ----
tata_sequences <- readDNAStringSet("attentions/test_datasets/custom_TATA_test.fa")
tata_sequences <- tibble(tag = str_split_i(names(tata_sequences), "\\|", 2),
                         name = paste0("sequence", seq(1, length(tata_sequences))),
                         sequence = as.character(tata_sequences))

## Fake TATA ----
fake_tata_sequences <- readDNAStringSet("attentions/test_datasets/fake_TATA_test.csv")
fake_tata_sequences <- tibble(tag = str_split_i(names(fake_tata_sequences), "\\|", 2),
                              name = paste0("sequence", seq(1, length(fake_tata_sequences))),
                              sequence = as.character(fake_tata_sequences))


## Enhancers ----
enhancer_sequences <- readDNAStringSet("attentions/test_datasets/enhancer_test.fa")
enhancer_sequences <- tibble(tag = str_split_i(names(enhancer_sequences), "\\|", 2),
                             name = paste0("sequence", seq(1, length(enhancer_sequences))),
                             sequence = as.character(enhancer_sequences)) %>% 
  dplyr::distinct(sequence, .keep_all = TRUE)



# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Processing raw files ----
## TATA ----
preprocessing_scores <- function(score.file, sequences){
  print(glue::glue("Processing file {score.file}"))
  qs.file <- score.file %>% str_replace("raw", "processed") %>% str_replace("\\.pkl", "\\.qs")
  pd$read_pickle(score.file) %>%
    map(bind_cols) %>%
    list_rbind(names_to = "sequence") %>%
    dplyr::left_join(sequences, by = "sequence") %>%
    dplyr::select(name, kmers, position, tag, starts_with('layer')) %>% 
    dplyr::mutate(across(where(~ is.numeric(.x)), ~ map_dbl(.x, ~ .x))) %>% 
    tidyr::pivot_longer(cols = starts_with('layer'),
                        names_to = 'id',
                        values_to = 'attention_scores') %>%
    tidyr::separate_wider_delim(cols = id, delim = "-", names = c("layer", "head")) %>%
    qsave(file = qs.file)
}
tata_files <- list.files(path = "files/raw",
                         pattern = "custom_tata",
                         full.names = TRUE)
map(tata_files, \(x) preprocessing_scores(x, sequences=tata_sequences))

## Fake TATA ----
preprocessing_scores <- function(score.file, sequences){
  print(glue::glue("Processing file {score.file}"))
  qs.file <- score.file %>% str_replace("raw", "processed") %>% str_replace("\\.pkl", "\\.qs")
  pd$read_pickle(score.file) %>%
    map(bind_cols) %>%
    list_rbind(names_to = "sequence") %>%
    dplyr::left_join(sequences, by = "sequence") %>%
    dplyr::select(name, kmers, tag, starts_with('layer')) %>% 
    dplyr::mutate(across(where(~ is.numeric(.x)), ~ map_dbl(.x, ~ .x))) %>% 
    tidyr::pivot_longer(cols = starts_with('layer'),
                        names_to = 'id',
                        values_to = 'attention_scores') %>%
    tidyr::separate_wider_delim(cols = id, delim = "-", names = c("layer", "head")) %>%
    qsave(file = qs.file)
}
fake_tata_files <- list.files(path = "files/raw",
                              pattern = "fake_tata",
                              full.names = TRUE)
map(fake_tata_files, \(x) preprocessing_scores(x, sequences=fake_tata_sequences))

## Enhancers ----
preprocessing_scores <- function(score.file, sequences){
  print(glue::glue("Processing file {score.file}"))
  qs.file <- score.file %>% str_replace("raw", "processed") %>% str_replace("\\.pkl", "\\.qs")
  pd$read_pickle(score.file) %>%
    map(bind_cols) %>%
    list_rbind(names_to = "sequence") %>%
    dplyr::left_join(sequences, by = "sequence") %>%
    dplyr::select(name, kmers, position, tag, starts_with('layer')) %>% 
    dplyr::mutate(across(where(~ is.numeric(.x)), ~ map_dbl(.x, ~ .x))) %>% 
    tidyr::pivot_longer(cols = starts_with('layer'),
                        names_to = 'id',
                        values_to = 'attention_scores') %>%
    tidyr::separate_wider_delim(cols = id, delim = "-", names = c("layer", "head")) %>%
    qsave(file = qs.file)
}
enhancer_files <- list.files(path = "files/raw",
                             pattern = "enhancer",
                             full.names = TRUE)
map(enhancer_files, \(x) preprocessing_scores(x, sequences=enhancer_sequences))

## scGPT ----
scgpt_files <- list.files(path = "files/raw",
                          pattern = "scgpt",
                          full.names = TRUE)
preprocessing_scores <- function(score.file){
  print(glue::glue("Processing file {score.file}"))
  qs.file <- score.file %>% str_replace("raw", "processed") %>% str_replace("\\.csv", "\\.qs")
  if (str_detect(score.file, "pancreas")){
    key_to_value <- c(
      "0"  = "MHC class II",
      "1"  = "PP",
      "2"  = "PSC",
      "3"  = "acinar",
      "4"  = "alpha",
      "5"  = "beta",
      "6"  = "delta",
      "7"  = "ductal",
      "8"  = "endothelial",
      "9"  = "epsilon",
      "10" = "macrophage",
      "11" = "mast",
      "12" = "schwann",
      "13" = "t_cell"
    )
    tmp <- read_delim(score.file,
                      delim = ";",
                      show_col_types = FALSE) %>% 
      dplyr::mutate(cell_id = glue::glue("cell{row_number()}")) %>% 
      dplyr::select(cell_id, gene_sequence, label, expression, starts_with('layer'))
  } else {
    key_to_value <- c(
      "0"  = "PVALB-expressing interneuron",
      "1"  = "SST-expressing interneuron",
      "2"  = "SV2C-expressing interneuron",
      "3"  = "VIP-expressing interneuron",
      "4"  = "astrocyte",
      "5"  = "cortical layer 2-3 excitatory neuron A",
      "6"  = "cortical layer 2-3 excitatory neuron B",
      "7"  = "cortical layer 4 excitatory neuron",
      "8"  = "cortical layer 5-6 excitatory neuron",
      "9"  = "endothelial cell",
      "10" = "microglial cell",
      "11" = "mixed excitatory neuron",
      "12" = "mixed glial cell?",
      "13" = "oligodendrocyte A",
      "14" = "oligodendrocyte C",
      "15" = "oligodendrocyte precursor cell",
      "16" = "phagocyte",
      "17" = "pyramidal neuron?"
    )
    tmp <- read_delim(score.file,
                      delim = ";",
                      show_col_types = FALSE) %>% 
      dplyr::mutate(cell_id = glue::glue("cell{row_number()}")) %>% 
      dplyr::select(cell_id, gene_sequence, label, expression, `GOCC envelope`, starts_with('layer'))
  }
  tmp %>% 
    tidyr::separate_longer_delim(cols = -label,
                                 delim = ",") %>% 
    dplyr::mutate(label = key_to_value[as.character(label)]) %>% 
    dplyr::filter(gene_sequence != "<cls>") %>% 
    tidyr::pivot_longer(cols = starts_with("layer"),
                        names_to = "layer",
                        values_to = "attention_scores") %>% 
    tidyr::separate_wider_delim(cols = layer,
                                delim = "_",
                                names = c("layer", "head")) %>% 
    dplyr::rename(gene = gene_sequence) %>% 
    qsave(file = qs.file)
}
map(scgpt_files, \(x) preprocessing_scores(x))


na <- read_delim('files/raw/scgpt_pancreas_random_init_scores.csv', delim = ";", show_col_types = FALSE)
table(na$label)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Loading processing files ----
## TATA ----
tata_files <- c("dnabert_ft" = "files/processed/DNABERT_custom_tata_scores_dictionary_attention_annotations.qs",
                "dnabert_pt" = "files/processed/DNABERT_custom_tata_scores_dictionary_attention_pretrained_annotations.qs",
                "dnabert_rpt" = "files/processed/DNABERT_custom_tata_scores_dictionary_attention_random_pretrained_annotations.qs",
                "dnabert_ri" = "files/processed/DNABERT_custom_tata_scores_dictionary_attention_random_init_annotations.qs",
                "nt_ft" = "files/processed/NT_custom_tata_scores_dictionary_attention_annotations.qs",
                "nt_pt" = "files/processed/NT_custom_tata_scores_dictionary_attention_pretrained_annotations.qs",
                "nt_ri" = "files/processed/NT_custom_tata_scores_dictionary_attention_random_init_annotations.qs")
tata_df_list <- set_names(names(tata_files)) %>% map(\(x) qread(tata_files[[x]]), .progress = TRUE)

## Fake TATA ----
fake_tata_files <- c("dnabert_ft" = "files/processed/DNABERT_fake_tata_scores_dictionary_attention.qs",
                     "dnabert_pt" = "files/processed/DNABERT_fake_tata_scores_dictionary_pretrained_attention.qs",
                     "dnabert_rpt" = "files/processed/DNABERT_fake_tata_scores_dictionary_random_pretrained_attention.qs",
                     "dnabert_ri" = "files/processed/DNABERT_fake_tata_scores_dictionary_random_init_attention.qs",
                     "nt_ft" = "files/processed/NT_fake_tata_scores_dictionary_attention.qs",
                     "nt_pt" = "files/processed/NT_fake_tata_scores_dictionary_pretrained_attention.qs",
                     "nt_ri" = "files/processed/NT_fake_tata_scores_dictionary_random_init_attention.qs")
fake_tata_df_list <- set_names(names(fake_tata_files)) %>% map(\(x) qread(fake_tata_files[[x]]), .progress = TRUE)

## Enhancers ----
enhancer_files <- c("dnabert_ft" = "files/processed/DNABERT_enhancer_scores_dictionary_attention_annotations.qs",
                    "dnabert_pt" = "files/processed/DNABERT_enhancer_scores_dictionary_attention_pretrained_annotations.qs",
                    "dnabert_rpt" = "files/processed/DNABERT_enhancer_scores_dictionary_attention_random_annotations.qs",
                    "dnabert_ri" = "files/processed/DNABERT_enhancer_scores_dictionary_attention_random_init_annotations.qs",
                    "nt_ft" = "files/processed/NT_enhancer_scores_dictionary_attention_annotations.qs",
                    "nt_pt" = "files/processed/NT_enhancer_scores_dictionary_pretrained_attention_annotations.qs",
                    "nt_ri" = "files/processed/NT_enhancer_scores_dictionary_random_init_attention_annotations.qs")
enhancer_df_list <- set_names(names(enhancer_files)) %>% map(\(x) qread(enhancer_files[[x]]), .progress = TRUE)

## scGPT ----
scgpt_files <- c("ms_ft" = "files/processed/scgpt_ms_finetuned_scores.qs",
                 "ms_pt" = "files/processed/scgpt_ms_pretrained_scores.qs",
                 "ms_ri" = "files/processed/scgpt_ms_random_init_scores.qs",
                 "pancreas_ft" = "files/processed/scgpt_pancreas_finetuned_scores.qs.qs",
                 "pancreas_pt" = "files/processed/scgpt_pancreas_pretrained_scores.qs.qs",
                 "pancreas_ri" = "files/processed/scgpt_pancreas_random_init_scores.qs.qs")
scgpt_df_list <- set_names(names(scgpt_files)) %>% map(\(x) qread(scgpt_files[[x]]), .progress = TRUE)


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Extract kmer information ----
## TATA ----
### Most activated ----
extract_activated_kmers <- function(x, n.slice=1000){
  # n.slice <- round(tata_df_list[[x]] %>% dplyr::filter(layer=="layer0" & head=="head0") %>% nrow() * p.slice)
  tata_df_list[[x]] %>% 
    dplyr::filter(nchar(kmers) == 6) %>% 
    dplyr::group_by(layer, head) %>%
    dplyr::slice_max(attention_scores, n=n.slice) %>%
    dplyr::ungroup() %>% 
    dplyr::select(name, kmers, position, layer, head, attention_scores)
} #10% most activated kmers: dnabert = 6254; nt = 1060
most_activated_kmers <- set_names(names(tata_files)) %>% map(\(x) extract_activated_kmers(x)) %>% 
  list_rbind(names_to = "model")
qsave(most_activated_kmers, file = 'files/intermediate/custom_tata_most_activated_kmers_all_models.qs')

### Abundance ----
kmer_abundance <- most_activated_kmers %>%
  dplyr::group_by(model, layer, head, kmers) %>%
  dplyr::summarise(n = n()) %>%
  dplyr::group_by(model, layer, head) %>%
  dplyr::mutate(perc = n/sum(n)*100) %>%
  dplyr::ungroup()
qsave(kmer_abundance, file = 'files/intermediate/custom_tata_most_activated_kmers_abundance_all_models.qs')

## Enhancer ----
### Most activated ----
extract_activated_kmers <- function(x, n.slice=1000){
  # n.slice <- round(enhancer_df_list[[x]] %>% dplyr::filter(layer=="layer0" & head=="head0") %>% nrow() * p.slice)
  enhancer_df_list[[x]] %>% 
    dplyr::filter(nchar(kmers) == 6) %>% 
    dplyr::group_by(layer, head) %>%
    dplyr::slice_max(attention_scores, n=n.slice) %>%
    dplyr::ungroup() %>% 
    dplyr::select(name, kmers, position, layer, head, attention_scores)
} #10% most activated kmers: dnabert = 5288; nt = 9386
most_activated_kmers <- set_names(names(enhancer_files)) %>% map(\(x) extract_activated_kmers(x)) %>% 
  list_rbind(names_to = "model")
qsave(most_activated_kmers, file = 'files/intermediate/enhancer_most_activated_kmers_all_models.qs')

### Abundance ----
kmer_abundance <- most_activated_kmers %>%
  dplyr::group_by(model, layer, head, kmers) %>%
  dplyr::summarise(n = n()) %>%
  dplyr::group_by(model, layer, head) %>%
  dplyr::mutate(perc = n/sum(n)*100) %>%
  dplyr::ungroup()
qsave(kmer_abundance, file = 'files/intermediate/enhancer_most_activated_kmers_abundance_all_models.qs')


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# Calculate by layer,head,tag metrics ----
## TATA ----
by_layer_head_tag_metrics <- function(x){
  print(glue::glue("Processing {x}"))
  
  threshold <- ifelse(startsWith(x, "dnabert"), 15, 3)
  scores <- tata_df_list[[x]]
  selected_kmers <- scores %>% 
    dplyr::filter(head == "head0" & layer == "layer0") %>% 
    dplyr::group_by(kmers, tag) %>% 
    dplyr::summarise(n = n(), .groups = "drop") %>%  
    tidyr::pivot_wider(names_from = tag,
                       values_from = n)  %>% 
    dplyr::rename(negative = `0`, positive = `1`) %>% 
    tidyr::drop_na() %>% 
    dplyr::filter(negative > threshold & positive > threshold) %>% 
    dplyr::filter(nchar(kmers) == 6) %>% 
    dplyr::pull(kmers)
  all_kmers_matrix <- scores %>% 
    dplyr::select(layer, head, tag, attention_scores) %>% 
    dplyr::group_by(layer, head, tag) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  by_kmers_matrix <- scores %>% 
    dplyr::select(layer, head, kmers, tag, attention_scores) %>% 
    dplyr::group_by(layer, head, kmers, tag) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  by_layer_head_kmers_matrix <- by_kmers_matrix %>% 
    dplyr::left_join(all_kmers_matrix %>%  dplyr::rename(control_scores = attention_scores),
                     by = c("layer", "head", "tag")) %>% 
    dplyr::mutate(difference = attention_scores-control_scores) %>% 
    dplyr::select(-attention_scores, -control_scores) %>% 
    dplyr::filter(kmers %in% selected_kmers) %>% 
    dplyr::mutate(id = glue::glue("{layer}_{head}_{kmers}")) %>% 
    tidyr::pivot_wider(names_from = tag,
                       values_from = difference) %>% 
    dplyr::rename(negative = `0`, positive = `1`) %>% 
    tidyr::drop_na() %>% 
    dplyr::mutate(distance_diag = abs(negative - positive)/sqrt(2))
  qsave(by_layer_head_kmers_matrix, file = glue::glue('files/intermediate/custom_tata_{x}_layer_head_tag_metrics.qs'))
}
set_names(names(tata_df_list)) %>% map(\(x) by_layer_head_tag_metrics(x))

## Fake TATA ----
by_layer_head_tag_metrics <- function(x){
  print(glue::glue("Processing {x}"))
  
  threshold <- ifelse(startsWith(x, "dnabert"), 15, 3)
  scores <- fake_tata_df_list[[x]]
  selected_kmers <- scores %>% 
    dplyr::filter(head == "head0" & layer == "layer0") %>% 
    dplyr::group_by(kmers, tag) %>% 
    dplyr::summarise(n = n(), .groups = "drop") %>%  
    tidyr::pivot_wider(names_from = tag,
                       values_from = n)  %>% 
    dplyr::rename(negative = `0`, positive = `1`) %>% 
    tidyr::drop_na() %>% 
    dplyr::filter(negative > threshold & positive > threshold) %>% 
    dplyr::filter(nchar(kmers) == 6) %>% 
    dplyr::pull(kmers)
  all_kmers_matrix <- scores %>% 
    dplyr::select(layer, head, tag, attention_scores) %>% 
    dplyr::group_by(layer, head, tag) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  by_kmers_matrix <- scores %>% 
    dplyr::select(layer, head, kmers, tag, attention_scores) %>% 
    dplyr::group_by(layer, head, kmers, tag) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  by_layer_head_kmers_matrix <- by_kmers_matrix %>% 
    dplyr::left_join(all_kmers_matrix %>%  dplyr::rename(control_scores = attention_scores),
                     by = c("layer", "head", "tag")) %>% 
    dplyr::mutate(difference = attention_scores-control_scores) %>% 
    dplyr::select(-attention_scores, -control_scores) %>% 
    dplyr::filter(kmers %in% selected_kmers) %>% 
    dplyr::mutate(id = glue::glue("{layer}_{head}_{kmers}")) %>% 
    tidyr::pivot_wider(names_from = tag,
                       values_from = difference) %>% 
    dplyr::rename(negative = `0`, positive = `1`) %>% 
    tidyr::drop_na() %>% 
    dplyr::mutate(distance_diag = abs(negative - positive)/sqrt(2))
  qsave(by_layer_head_kmers_matrix, file = glue::glue('files/intermediate/fake_tata_{x}_layer_head_tag_metrics.qs'))
}
set_names(names(fake_tata_df_list)) %>% map(\(x) by_layer_head_tag_metrics(x))

## Enhancer ----
by_layer_head_tag_metrics <- function(x){
  print(glue::glue("Processing {x}"))
  
  threshold <- ifelse(startsWith(x, "dnabert"), 15, 3)
  scores <- enhancer_df_list[[x]]
  selected_kmers <- scores %>% 
    dplyr::filter(head == "head0" & layer == "layer0") %>% 
    dplyr::group_by(kmers, tag) %>% 
    dplyr::summarise(n = n(), .groups = "drop") %>%  
    tidyr::pivot_wider(names_from = tag,
                       values_from = n)  %>% 
    dplyr::rename(negative = `0`, positive = `1`) %>% 
    tidyr::drop_na() %>% 
    dplyr::filter(negative > threshold & positive > threshold) %>% 
    dplyr::filter(nchar(kmers) == 6) %>% 
    dplyr::pull(kmers)
  all_kmers_matrix <- scores %>% 
    dplyr::select(layer, head, tag, attention_scores) %>% 
    dplyr::group_by(layer, head, tag) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  by_kmers_matrix <- scores %>% 
    dplyr::select(layer, head, kmers, tag, attention_scores) %>% 
    dplyr::group_by(layer, head, kmers, tag) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  by_layer_head_kmers_matrix <- by_kmers_matrix %>% 
    dplyr::left_join(all_kmers_matrix %>%  dplyr::rename(control_scores = attention_scores),
                     by = c("layer", "head", "tag")) %>% 
    dplyr::mutate(difference = attention_scores-control_scores) %>% 
    dplyr::select(-attention_scores, -control_scores) %>% 
    dplyr::filter(kmers %in% selected_kmers) %>% 
    dplyr::mutate(id = glue::glue("{layer}_{head}_{kmers}")) %>% 
    tidyr::pivot_wider(names_from = tag,
                       values_from = difference) %>% 
    dplyr::rename(negative = `0`, positive = `1`) %>% 
    tidyr::drop_na() %>% 
    dplyr::mutate(distance_diag = abs(negative - positive)/sqrt(2))
  qsave(by_layer_head_kmers_matrix, file = glue::glue('files/intermediate/enhancer_{x}_layer_head_tag_metrics.qs'))
}
set_names(names(enhancer_df_list)) %>% map(\(x) by_layer_head_tag_metrics(x))

## scGPT by cell type ----
by_layer_head_tag_metrics <- function(x){
  print(glue::glue("Processing {x}"))
  
  if (str_detect(x, "pancreas")){
    scores <- scgpt_df_list[[x]] %>% 
      dplyr::mutate(label = ifelse(label == "beta",
                                   "selection",
                                   "other"),
                    attention_scores = as.numeric(attention_scores))
  } else {
    scores <- scgpt_df_list[[x]] %>% 
      dplyr::mutate(label = ifelse(label == "cortical layer 2-3 excitatory neuron A",
                                   "selection",
                                   "other"),
                    attention_scores = as.numeric(attention_scores))
  }
  
  selected_genes <- scores %>% 
    dplyr::filter(head == "head0" & layer == "layer0") %>% 
    dplyr::group_by(gene, label) %>% 
    dplyr::summarise(n = n(), .groups = "drop") %>%  
    tidyr::pivot_wider(names_from = label,
                       values_from = n)  %>% 
    tidyr::drop_na() %>% 
    dplyr::filter(selection > 15 & other > 15) %>% 
    dplyr::pull(gene)
  all_genes_matrix <- scores %>% 
    dplyr::select(layer, head, label, attention_scores) %>% 
    dplyr::group_by(layer, head, label) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  by_genes_matrix <- scores %>% 
    dplyr::select(layer, head, gene, label, attention_scores) %>% 
    dplyr::group_by(layer, head, gene, label) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  by_layer_head_genes_matrix <- by_genes_matrix %>% 
    dplyr::left_join(all_genes_matrix %>%  dplyr::rename(control_scores = attention_scores),
                     by = c("layer", "head", "label")) %>% 
    dplyr::mutate(difference = attention_scores-control_scores) %>% 
    dplyr::select(-attention_scores, -control_scores) %>% 
    dplyr::filter(gene %in% selected_genes) %>% 
    dplyr::mutate(id = glue::glue("{layer}_{head}_{gene}")) %>% 
    tidyr::pivot_wider(names_from = label,
                       values_from = difference) %>% 
    tidyr::drop_na() %>% 
    dplyr::mutate(distance_diag = abs(other - selection)/sqrt(2))
  qsave(by_layer_head_genes_matrix, file = glue::glue('files/intermediate/scgpt_{x}_layer_head_tag_metrics.qs'))
}
set_names(names(scgpt_df_list)) %>% map(\(x) by_layer_head_tag_metrics(x))

## scGPT by gene expression ----
by_layer_head_tag_metrics <- function(x){
  print(glue::glue("Processing {x}"))
  
  gene_expression <- scgpt_df_list[[x]] %>% 
    dplyr::select(cell_id, gene, expression) %>% 
    dplyr::distinct() %>% 
    dplyr::mutate(expression = as.numeric(expression)) %>% 
    dplyr::group_by(gene) %>% 
    dplyr::summarise(mean = mean(expression))
  scores <- scgpt_df_list[[x]] %>% 
    dplyr::mutate(expression = as.numeric(expression),
                  attention_scores = as.numeric(attention_scores)) %>% 
    dplyr::left_join(gene_expression, by = "gene") %>% 
    dplyr::mutate(exp_level = ifelse(expression <= mean,
                                     "low",
                                     "high")) %>% 
    dplyr::select(-expression, -mean)
  selected_genes <- scores %>% 
    dplyr::filter(head == "head0" & layer == "layer0") %>% 
    dplyr::group_by(gene, exp_level) %>% 
    dplyr::summarise(n = n(), .groups = "drop") %>%  
    tidyr::pivot_wider(names_from = exp_level,
                       values_from = n)  %>% 
    tidyr::drop_na() %>% 
    dplyr::filter(low > 10 & high > 10) %>% 
    dplyr::pull(gene)
  all_genes_matrix <- scores %>% 
    dplyr::select(layer, head, exp_level, attention_scores) %>% 
    dplyr::group_by(layer, head, exp_level) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  by_genes_matrix <- scores %>% 
    dplyr::select(layer, head, gene, exp_level, attention_scores) %>% 
    dplyr::group_by(layer, head, gene, exp_level) %>% 
    dplyr::summarise(attention_scores = mean(attention_scores), .groups = "drop")
  by_layer_head_genes_matrix <- by_genes_matrix %>% 
    dplyr::left_join(all_genes_matrix %>%  dplyr::rename(control_scores = attention_scores),
                     by = c("layer", "head", "exp_level")) %>% 
    dplyr::mutate(difference = attention_scores-control_scores) %>% 
    dplyr::select(-attention_scores, -control_scores) %>% 
    dplyr::filter(gene %in% selected_genes) %>% 
    dplyr::mutate(id = glue::glue("{layer}_{head}_{gene}")) %>% 
    tidyr::pivot_wider(names_from = exp_level,
                       values_from = difference) %>% 
    tidyr::drop_na() %>% 
    dplyr::mutate(distance_diag = abs(low - high)/sqrt(2))
  qsave(by_layer_head_genes_matrix, file = glue::glue('files/intermediate/scgpt_{x}_layer_head_exp_level_metrics.qs'))
}
set_names(names(scgpt_df_list)) %>% map(\(x) by_layer_head_tag_metrics(x))
