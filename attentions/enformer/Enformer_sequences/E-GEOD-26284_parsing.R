library(data.table)
library(dplyr)
library(tidyr)
library(ensembldb)
library(AnnotationHub)

# Download Ensembl HG38 database
ahDb <- AnnotationHub()
edb <- ahDb[["AH109606"]] #hg38

# Load original data
df <- fread("/Enformer/enformer_sequences/E-GEOD-26284-query-results.tpms.tsv", skip = 4) %>% 
  as_tibble() %>% 
  dplyr::select(`Gene ID`, `Gene Name`, `whole cell, long polyA RNA, K562`) %>% 
  dplyr::rename(GeneID = `Gene ID`,
                GeneName = `Gene Name`,
                TPMS = `whole cell, long polyA RNA, K562`) %>% 
  dplyr::filter(TPMS > 5)
df

# Extract genomic positions
gene.loc <- genes(edb, filter = c(GeneNameFilter(df$GeneName), GeneBiotypeFilter("protein_coding"))) %>% as_tibble()
transcript.loc <- transcripts(edb, filter = TxNameFilter(gene.loc$canonical_transcript)) %>% 
  as_tibble() %>% 
  dplyr::left_join(., gene.loc %>% dplyr::select(gene_id, gene_name, start) %>% dplyr::rename(gene_start = start), by = "gene_id") %>% 
  dplyr::group_by(seqnames) %>% 
  dplyr::mutate(distance = gene_start-lag(gene_start, default = 0),
                distance = ifelse((distance == gene_start),
                                  0,
                                  distance)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(lonely = ifelse((distance > 197000) & (lead(distance, default = 0) > 197000),
                                "Yes",
                                "No"),
                seqnames = paste0("chr", seqnames)) %>% 
  dplyr::select(gene_name, gene_id, tx_id, seqnames, start, end, strand, distance, lonely) %>% 
  dplyr::rename(chr = seqnames)
transcript.loc

write.table(transcript.loc %>% dplyr::filter(lonely == "Yes") %>% dplyr::arrange(desc(distance)), file = "/Enformer/enformer_sequences/K562_expressed_genes_lonely_197000.tsv", row.names = FALSE, quote = FALSE)
