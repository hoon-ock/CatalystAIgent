import numpy as np
from ase import neighborlist, Atoms
from ase.neighborlist import natural_cutoffs
from scipy.spatial.distance import euclidean
from itertools import combinations 

class SiteAnalyzer:
    def __init__(self, adslab, cutoff_multiplier=1.0):
        """
        Initialize class to handle site based analysis.        
        Args:
            adslab (ase.Atoms): object of the slab with the adsorbate placed.
        """
        self.atoms = adslab
        self.cutoff_multiplier = cutoff_multiplier
        self.binding_info = self._find_binding_graph()
        self.second_binding_info = self._find_second_binding_graph()
        self.center_atom = self._get_center_ads_atom()
        self.center_binding_info = self._find_binding_atoms_from_center()    

    def _get_center_ads_atom(self):
        """
        Identify the center atom in adsorbate
        """
        tags = self.atoms.get_tags()
        elements = self.atoms.get_chemical_symbols()
        adsorbate_atom_idxs = [idx for idx, tag in enumerate(tags) if tag == 2]
        adsorbate_atom_positions = self.atoms.get_positions()[adsorbate_atom_idxs]
        center_position = np.mean(adsorbate_atom_positions, axis=0)
        all_distance = [euclidean(center_position, position) for position in adsorbate_atom_positions]
        min_idx = adsorbate_atom_idxs[np.argmin(all_distance)]
        center_atom = self.atoms[min_idx]        
        return center_atom    

    def _find_binding_atoms_from_center(self):
        """
        Find the binding atoms from the center atom
        """
        tags = self.atoms.get_tags()
        elements = self.atoms.get_chemical_symbols()
        # slab_atom --> surface_atom
        slab_atom_idxs = [idx for idx, tag in enumerate(tags) if tag == 1]
        # breakpoint()
        # convert Atom object to Atoms object which is iterable
        # center_atom = Atoms()
        # center_atom.append(self.center_atom)
        connectivity = self._get_connectivity(self.atoms, self.cutoff_multiplier)
        binding_info = []
        adslab_positions = self.atoms.get_positions()
        # breakpoint()
        if sum(connectivity[self.center_atom.index][slab_atom_idxs]) >= 1:
            bound_slab_idxs = [
                idx_slab
                for idx_slab in slab_atom_idxs
                if connectivity[self.center_atom.index][idx_slab] == 1
            ]
            ads_idx_info = {
                "adsorbate_binding_atom_indices": self.center_atom.index,
                "adsorbate_binding_atom_elements": elements[self.center_atom.index],
                "surface_binding_atom_elements": [
                    element
                    for idx_el, element in enumerate(elements)
                    if idx_el in bound_slab_idxs
                ],
                "surface_binding_atom_indices": bound_slab_idxs,
                "binding_positions": adslab_positions[self.center_atom.index],
            }
            binding_info.append(ads_idx_info)
        return binding_info    

    def _find_second_binding_graph(self):
        tags = self.atoms.get_tags()
        elements = self.atoms.get_chemical_symbols()        
        connectivity = self._get_connectivity(self.atoms, self.cutoff_multiplier)        
        second_binding_info = {}
        for interaction in self.binding_info:
            slab_atom_idxs = interaction["surface_binding_atom_indices"]
            if len(slab_atom_idxs) != 0:
                for idx in slab_atom_idxs:
                    # import pdb; pdb.set_trace()
                    second_connection = connectivity[idx] == 1
                    second_int_idx = np.where(second_connection)[0]
                    second_int_info = {
                                        #"slab_idx": idx,
                                        "slab_element": elements[idx],
                                        "second_interaction_idx": second_int_idx,
                                        "second_interaction_element": [elements[idx] for idx in second_int_idx]
                                        }
                    second_binding_info.update({idx: second_int_info}) #.append(second_int_info)
        return second_binding_info    

    def _find_binding_graph(self):
        tags = self.atoms.get_tags()
        elements = self.atoms.get_chemical_symbols()        
        adsorbate_atom_idxs = [idx for idx, tag in enumerate(tags) if tag == 2]
        slab_atom_idxs = [idx for idx, tag in enumerate(tags) if tag != 2]        
        connectivity = self._get_connectivity(self.atoms, self.cutoff_multiplier)        
        binding_info = []
        adslab_positions = self.atoms.get_positions()
        for idx in adsorbate_atom_idxs:
            if sum(connectivity[idx][slab_atom_idxs]) >= 1:
                bound_slab_idxs = [
                    idx_slab
                    for idx_slab in slab_atom_idxs
                    if connectivity[idx][idx_slab] == 1
                ]
                ads_idx_info = {
                    "adsorbate_binding_atom_indices": idx,
                    "adsorbate_binding_atom_elements": elements[idx],
                    "surface_binding_atom_elements": [
                        element
                        for idx_el, element in enumerate(elements)
                        if idx_el in bound_slab_idxs
                    ],
                    "surface_binding_atom_indices": bound_slab_idxs,
                    "binding_positions": adslab_positions[idx],
                }
                binding_info.append(ads_idx_info)
        return binding_info    

    def _get_connectivity(self, atoms, cutoff_multiplier=1.0):
        """
        Note: need to condense this with the surface method
        Generate the connectivity of an atoms obj.
        Args:
            atoms (ase.Atoms): object which will have its connectivity considered
            cutoff_multiplier (float, optional): cushion for small atom movements when assessing
                atom connectivity
        Returns:
            (np.ndarray): The connectivity matrix of the atoms object.
        """
        cutoff = natural_cutoffs(atoms, mult=cutoff_multiplier)
        neighbor_list = neighborlist.NeighborList(
            cutoff,
            self_interaction=False,
            bothways=True,
            skin=0.05,
        )
        neighbor_list.update(atoms)
        matrix = neighborlist.get_connectivity_matrix(neighbor_list.nl).toarray()
        return matrix    

    def get_dentate(self):
        """
        Get the number of adsorbate atoms that are bound to the surface.       
        Returns:
            (int): The number of binding interactions
        """
        return len(self.binding_info)    

    def get_site_types(self):
        """
        Get the number of surface atoms the bound adsorbate atoms are interacting with as a
        proximate for hollow, bridge, and atop binding.        
        Returns:
            (list[int]): number of interacting surface atoms for each adsorbate atom bound.
        """
        return [len(binding["surface_binding_atom_indices"]) for binding in self.binding_info]
    
    def get_center_site_type(self):
        """
        Get the number of surface atoms the bound adsorbate center atom is interacting with as a
        proximate for hollow, bridge, and atop binding.        
        Returns:
            (list[int]): number of interacting surface atoms for each adsorbate atom bound.
        """
        return [len(binding["surface_binding_atom_indices"]) for binding in self.center_binding_info]


    def get_bound_atom_positions(self):
        """
        Get the euclidean coordinates of all bound adsorbate atoms.        
        Returns:
            (list[np.array]): euclidean coordinates of bound atoms
        """
        positions = []
        for atom in self.binding_info:
            positions.append(atom["binding_positions"])
        return positions    

    def get_minimum_site_proximity(self, site_to_compare):
        """
        Note: might be good to check the surfaces are identical and raise an error otherwise.
        Get the minimum distance between bound atoms on the surface between two adsorbates.        
        Args:
            site_to_compare (catapalt.SiteAnalyzer): site analysis instance of the other adslab.        
        Returns:
            (float): The minimum distance between bound adsorbate atoms on a surface.
                and returns `np.nan` if one or more of the configurations was not
                surface bound.
        """
        this_positions = self.get_bound_atom_positions()
        other_positions = site_to_compare.get_bound_atom_positions()
        distances = []
        if len(this_positions) > 0 and len(other_positions) > 0:
            for this_position in this_positions:
                for other_position in other_positions:
                    fake_atoms = Atoms("CO", positions=[this_position, other_position])
                    distances.append(fake_atoms.get_distance(0, 1, mic=True))
            return min(distances)
        else:
            return np.nan    

    def get_adsorbate_bond_lengths(self):
        """ """
        bond_lengths = {"adsorbate-adsorbate": {}, "adsorbate-surface": {}}
        adsorbate = self.atoms[
            [idx for idx, tag in enumerate(self.atoms.get_tags()) if tag == 2]
        ]
        adsorbate_connectivity = self._get_connectivity(adsorbate)
        combos = list(combinations(range(len(adsorbate)), 2))
        for combo in combos:
            if adsorbate_connectivity[combo[0], combo[1]] == 1:
                bond_lengths["adsorbate-adsorbate"][
                    tuple(np.sort(combo))
                ] = adsorbate.get_distance(combo[0], combo[1], mic=True)        
        for ads_info in self.binding_info:
            adsorbate_idx = ads_info["adsorbate_binding_atom_indices"]
            bond_lengths["adsorbate-surface"][adsorbate_idx] = [
                self.atoms.get_distances(adsorbate_idx, slab_idx, mic=True)[0]
                for slab_idx in ads_info["surface_binding_atom_indices"]
            ]        
        return bond_lengths
    

class DetectTrajAnomaly:
    def __init__(
        self,
        init_atoms,
        final_atoms,
        atoms_tag,
        final_slab_atoms=None,
        surface_change_cutoff_multiplier=1.5,
        desorption_cutoff_multiplier=1.5,
    ):
        """
        Flag anomalies based on initial and final stucture of a relaxation.
        Args:
            init_atoms (ase.Atoms): the adslab in its initial state
            final_atoms (ase.Atoms): the adslab in its final state
            atoms_tag (list): the atom tags; 0=bulk, 1=surface, 2=adsorbate
            final_slab_atoms (ase.Atoms, optional): the relaxed slab if unspecified this defaults
            to using the initial adslab instead.
            surface_change_cutoff_multiplier (float, optional): cushion for small atom movements
                when assessing atom connectivity for reconstruction
            desorption_cutoff_multiplier (float, optional): cushion for physisorbed systems to not
                be discarded. Applied to the covalent radii.
        """
        self.init_atoms = init_atoms
        self.final_atoms = final_atoms
        self.final_slab_atoms = final_slab_atoms
        self.atoms_tag = atoms_tag
        self.surface_change_cutoff_multiplier = surface_change_cutoff_multiplier
        self.desorption_cutoff_multiplier = desorption_cutoff_multiplier

        if self.final_slab_atoms is None:
            slab_idxs = [idx for idx, tag in enumerate(self.atoms_tag) if tag != 2]
            self.final_slab_atoms = self.init_atoms[slab_idxs]

    def is_adsorbate_dissociated(self):
        """
        Tests if the initial adsorbate connectivity is maintained.
        Returns:
            (bool): True if the connectivity was not maintained, otherwise False
        """
        adsorbate_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag == 2]
        return not (
            np.array_equal(
                self._get_connectivity(self.init_atoms[adsorbate_idx]),
                self._get_connectivity(self.final_atoms[adsorbate_idx]),
            )
        )

    def has_surface_changed(self):
        """
        Tests bond breaking / forming events within a tolerance on the surface so
        that systems with significant adsorbate induced surface changes may be discarded
        since the reference to the relaxed slab may no longer be valid.
        Returns:
            (bool): True if the surface is reconstructed, otherwise False
        """
        surf_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag != 2]

        adslab_connectivity = self._get_connectivity(self.final_atoms[surf_idx])
        slab_connectivity_w_cushion = self._get_connectivity(
            self.final_slab_atoms, self.surface_change_cutoff_multiplier
        )
        slab_test = 1 in (adslab_connectivity - slab_connectivity_w_cushion)

        adslab_connectivity_w_cushion = self._get_connectivity(
            self.final_atoms[surf_idx], self.surface_change_cutoff_multiplier
        )
        slab_connectivity = self._get_connectivity(self.final_slab_atoms)
        adslab_test = 1 in (slab_connectivity - adslab_connectivity_w_cushion)
        # breakpoint()
        return any([slab_test, adslab_test])

    def is_adsorbate_desorbed(self):
        """
        If the adsorbate binding atoms have no connection with slab atoms,
        consider it desorbed.
        Returns:
            (bool): True if there is desorption, otherwise False
        """
        adsorbate_atoms_idx = [
            idx for idx, tag in enumerate(self.atoms_tag) if tag == 2
        ]
        slab_atoms_idx = [idx for idx, tag in enumerate(self.atoms_tag) if tag != 2]
        final_connectivity = self._get_connectivity(
            self.final_atoms, self.desorption_cutoff_multiplier
        )

        for idx in adsorbate_atoms_idx:
            if sum(final_connectivity[idx][slab_atoms_idx]) >= 1:
                return False
        return True

    def _get_connectivity(self, atoms, cutoff_multiplier=1.0):
        """
        Generate the connectivity of an atoms obj.
        Args:
            atoms (ase.Atoms): object which will have its connectivity considered
            cutoff_multiplier (float, optional): cushion for small atom movements when assessing
                atom connectivity
        Returns:
            (np.ndarray): The connectivity matrix of the atoms object.
        """
        cutoff = natural_cutoffs(atoms, mult=cutoff_multiplier)
        neighbor_list = neighborlist.NeighborList(
            cutoff, self_interaction=False, bothways=True
        )
        neighbor_list.update(atoms)
        matrix = neighborlist.get_connectivity_matrix(neighbor_list.nl).toarray()
        return matrix