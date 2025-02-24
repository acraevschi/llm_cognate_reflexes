import subprocess

# Doesn't work yet


def get_phylogenetic_tree(glottocodes):
    """
    Given a list of Glottocodes, call the R script to extract the phylogenetic tree in Newick format.

    Parameters:
    glottocodes (list): A list of Glottocode strings.

    Returns:
    str: Phylogenetic tree in Newick format.
    """
    # Join the list of Glottocodes into a comma-separated string
    glottocodes_str = ",".join(glottocodes)

    # Define the command to call the R script
    command = [
        "Rscript",
        "glotto_trees/extract_tree.R",
        glottocodes_str,
    ]

    try:
        # Execute the R script and capture the output
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, shell=True
        )
        newick_tree = result.stdout.strip()
        return newick_tree
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while calling the R script: {e}")
        return None
