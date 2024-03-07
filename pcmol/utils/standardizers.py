from typing import Literal
from papyrus_structure_pipeline import standardizer as Papyrus_standardizer
from papyrus_structure_pipeline.standardizer import StandardizationResult
from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import FragmentParent

"""
From chemstore package by Martin Sicho
Originally created by Olivier Bequignon:
    https://github.com/OlivierBeq/Papyrus_structure_pipeline

Used for standardizing the generated molecules.
"""


class PapyrusStandardizer:
    def __init__(
        self,
        keep_stereo: bool = True,
        canonize: bool = True,
        mixture_handling: Literal["keep_largest", "filter", "keep"] = "keep_largest",
        remove_additional_salts: bool = True,
        remove_additional_metals: bool = True,
        filter_inorganic: bool = False,
        filter_non_small_molecule: bool = True,
        small_molecule_min_mw: float = 200,
        small_molecule_max_mw: float = 800,
        canonicalize_tautomer: bool = True,
        tautomer_max_tautomers: int = 2**32 - 1,
        extra_organic_atoms: list = None,
        extra_metals: list = None,
        extra_salts: list = None,
    ):
        self._settings = {
            "keep_stereo": keep_stereo,
            "canonize": canonize,
            "remove_additional_salts": remove_additional_salts,
            "remove_additional_metals": remove_additional_metals,
            "filter_inorganic": filter_inorganic,
            "filter_non_small_molecule": filter_non_small_molecule,
            "canonicalize_tautomer": canonicalize_tautomer,
            "small_molecule_min_mw": small_molecule_min_mw,
            "small_molecule_max_mw": small_molecule_max_mw,
            "tautomer_allow_stereo_removal": not keep_stereo,
            "tautomer_max_tautomers": tautomer_max_tautomers,
            "extra_organic_atoms": (
                sorted(extra_organic_atoms) if extra_organic_atoms else []
            ),
            "extra_metals": sorted(extra_metals) if extra_metals else [],
            "extra_salts": sorted(extra_salts) if extra_salts else [],
            "mixture_handling": mixture_handling,
        }
        if self._settings["extra_organic_atoms"]:
            Papyrus_standardizer.ORGANIC_ATOMS.extend(
                self._settings["extra_organic_atoms"]
            )
        if self._settings["extra_metals"]:
            Papyrus_standardizer.METALS.extend(self._settings["extra_metals"])
        if self._settings["extra_salts"]:
            Papyrus_standardizer.SALTS.extend(self._settings["extra_salts"])

    def __call__(self, smiles):
        return self.convertSMILES(smiles)

    @classmethod
    def fromSettingsFile(cls, path: str):
        """
        Load the standardizer from a settings file in JSON format.

        :param path:
        :return:
        """
        import json

        with open(path, "r") as f:
            settings = json.load(f)
        return cls.fromSettings(settings)

    def fix_errors(self, mol, error):
        if (
            error == StandardizationResult.MIXTURE_MOLECULE
            and self._settings["mixture_handling"] == "keep_largest"
        ):
            mol = FragmentParent(mol)
            return mol

        return None

    def convertSMILES(self, smiles, verbose=False):
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        out = Papyrus_standardizer.standardize(
            mol,
            return_type=True,
            remove_additional_salts=self._settings["remove_additional_salts"],
            remove_additional_metals=self._settings["remove_additional_metals"],
            filter_mixtures=(
                False if self._settings["mixture_handling"] == "keep" else True
            ),
            filter_inorganic=self._settings["filter_inorganic"],
            filter_non_small_molecule=self._settings["filter_non_small_molecule"],
            small_molecule_min_mw=self._settings["small_molecule_min_mw"],
            small_molecule_max_mw=self._settings["small_molecule_max_mw"],
            canonicalize_tautomer=self._settings["canonicalize_tautomer"],
            tautomer_max_tautomers=self._settings["tautomer_max_tautomers"],
            tautomer_allow_stereo_removal=self._settings[
                "tautomer_allow_stereo_removal"
            ],
        )
        results = [x for x in out[1:]]
        if not StandardizationResult.CORRECT_MOLECULE in results:
            mol = self.fix_errors(mol, results[-1])
            if not mol:
                if verbose:
                    print("SMILES rejected", smiles)
                    print("\tCause:", results)
                return None, smiles
            else:
                return (
                    self.convertSMILES(
                        Chem.MolToSmiles(
                            mol,
                            isomericSmiles=self._settings["keep_stereo"],
                            canonical=self._settings["canonize"],
                        )
                    ),
                    smiles,
                )
        else:
            return (
                Chem.MolToSmiles(
                    out[0],
                    canonical=self._settings["canonize"],
                    isomericSmiles=self._settings["keep_stereo"],
                )
                if out[0]
                else None
            ), smiles

    @property
    def settings(self):
        return self._settings

    def getID(self):
        sorted_keys = sorted(self._settings.keys())
        return "PapyrusStandardizer~" + ":".join(
            [f"{key}={str(self._settings[key])}" for key in sorted_keys]
        )

    def fromSettings(self, settings: dict):
        return PapyrusStandardizer(**settings)

    def getHashID(self):
        import hashlib

        return hashlib.md5(self.getID()).hexdigest()
