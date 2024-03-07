class ProteinSequence:
    """
    Simple object for storing protein info
    """

    def __init__(self, sequence=None, protein_id=None, filename=None, info=None):
        self.info = info
        self.seq = sequence
        self.protein_id = protein_id
        self.len = None
        self.ligand_idxs = []  # Datapoint line idxs in papyrus
        self.n_ligands = 0  # Number of datapoints in papyrus

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return f"Protein Seq: {self.protein_id}, length: \
                {len(self)}, ligands: {self.n_ligands}"
