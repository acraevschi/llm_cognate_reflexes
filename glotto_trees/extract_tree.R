# Doesn't work yet

# Load necessary libraries
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}

if (!requireNamespace("glottoTrees", quietly = TRUE)) {
  devtools::install_github("erichround/glottoTrees")
}

library(glottoTrees)
library(ape)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
    stop("No Glottocodes provided")
}
glottocodes <- unlist(strsplit(args[1], ","))

# Function to extract phylogenetic tree
extract_tree <- function(glottocodes) {
  # Retrieve the phylogenetic tree from Glottolog data
  glottolog_tree <- glottoTrees::get_glottolog_trees()
  glottolog_tree <- glottoTrees::assemble_rake(glottolog_tree)
  glottolog_tree <- glottoTrees::abridge_labels(glottolog_tree)

  # Extract the subtree containing the specified Glottocodes
  subtree <- ape::keep.tip(glottolog_tree, glottocodes)

  # Return the subtree in Newick format
  return(ape::write.tree(subtree))
}

# Extract and print the tree
cat(extract_tree(glottocodes))
