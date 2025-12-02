import numpy as np
import sys
import quantnmr.md as qnmd
from MDAnalysis.analysis import dihedrals


# Bond and angle definitions

bond_def = {
    'ALA_CB': ['name CB', 'name CA'],
    'ILE_CD': ['name CD', 'name CG1'],
    'ILE_CG2': ['name CG2', 'name CB'],
    'LEU_CD1': ['name CD1', 'name CG'],
    'LEU_proR': ['name CD1', 'name CG'],
    'LEU_CD2': ['name CD2', 'name CG'],
    'LEU_proS': ['name CD2', 'name CG'],
    'MET_CE': ['name SD', 'name CE'],
    'VAL_CG1': ['name CG1', 'name CB'],
    'VAL_proR': ['name CG1', 'name CB'],
    'VAL_CG2': ['name CG2', 'name CB'],
    'VAL_proS': ['name CG2', 'name CB'],
}   

phi_def = {
    #'ALA_CB': '(name CB or name CA or name C or name O)',
    'ALA_CB': '(name CB or name CA or name N or name HN)',
    'ILE_CD': '(name CD or name CG1 or name CB or name CA)',
    #'ILE_CG2': '(name CG2 or name CB or name CA or name C)',
    'ILE_CG2': '(name CG2 or name CB or name CA or name N)',
    'LEU_CD1': '(name CD1 or name CG or name CB or name CA)',
    'LEU_proR': '(name CD1 or name CG or name CB or name CA)',
    'LEU_CD2': '(name CD2 or name CG or name CB or name CA)',
    'LEU_proS': '(name CD2 or name CG or name CB or name CA)',
    'MET_CE': '(name CB or name CG or name SD or name CE)',
    # 'VAL_CG1': '(name CG1 or name CB or name CA or name C)',
    # 'VAL_proR': '(name CG1 or name CB or name CA or name C)',
    # 'VAL_CG2': '(name CG2 or name CB or name CA or name C)',
    # 'VAL_proS': '(name CG2 or name CB or name CA or name C)',
    'VAL_CG1': '(name CG1 or name CB or name CA or name N)',
    'VAL_proR': '(name CG1 or name CB or name CA or name N)',
    'VAL_CG2': '(name CG2 or name CB or name CA or name N)',
    'VAL_proS': '(name CG2 or name CB or name CA or name N)'
}   

theta_def = {
    #'ALA_CB': '(name CB or name CA or name C)',
    'ALA_CB': '(name CB or name CA or name N)',
    'ILE_CD': '(name CD or name CG1 or name CB)',           
    'ILE_CG2': '(name CG2 or name CB or name CA)',           
    'LEU_CD1': '(name CD1 or name CG or name CB)',           
    'LEU_proR': '(name CD1 or name CG or name CB)',           
    'LEU_CD2': '(name CD2 or name CG or name CB)',           
    'LEU_proS': '(name CD2 or name CG or name CB)',  
    'MET_CE': '(name CG or name SD or name CE)',
    'VAL_CG1': '(name CG1 or name CB or name CA)',           
    'VAL_proR': '(name CG1 or name CB or name CA)',           
    'VAL_CG2': '(name CG2 or name CB or name CA)',           
    'VAL_proS': '(name CG2 or name CB or name CA)',
}    


def O2_from_subtrajectory_by_bond_vector(trajectory_list: list, system_dict: dict, factor: int=1):
    '''
    Calculate the O2 order parameter from subtrajectories of a trajectory list.
    The subtrajectories are defined by the system_dict, which is a dictionary of lists of systems
    factor is the number of subtrajectories to take from the original trajectory

    '''
    len_traj = len(trajectory_list[0].trajectory) # we assume all the trajectories in the list are the same length
    span = len_traj//factor # we want minimum integer steps here e.g. 5001//10 = 500 (int) - length of traj to consider is 500

    O2_dict = {} # we define a dictionary to store and return results

    for aa_type in system_dict:
        selection_string = f'resname {aa_type}'
        selection_aa = trajectory_list[0].select_atoms(selection_string)
        aa_idx = sorted(set(selection_aa.resids)) # reduces multiple atom selections down to just the residue ID numbers
        print(aa_type, aa_idx)

        for methyl in system_dict[aa_type]:
            aa_methyl = f'{aa_type}_{methyl}'
            if aa_methyl not in bond_def:
                print(f'Error - did not select a valid methyl group - you should stop and fix this: {aa_methyl}?')
            else:
                # calc O2
                O2_bond = np.zeros((len(trajectory_list)*factor, len(aa_idx)))
                for i in range(len(trajectory_list)):
                    for nj, j in enumerate(aa_idx):
                        selection_0 = trajectory_list[i].select_atoms(f'resid {j} and {bond_def[aa_methyl][0]}')
                        selection_1 = trajectory_list[i].select_atoms(f'resid {j} and {bond_def[aa_methyl][1]}')
                        bond_vectors_normal = np.zeros((span, 3)) # matrix for normalized bond vectors in order times (5000ish), normalized bond vector (3)
                        for k in range(factor):
                            #fiter = t_sys[i].trajectory[k*span:(k+1)*span]
                            #frames = [ts for ts in fiter]
                            #print(k*span, (k+1)*span)
                            en3 = 0
                            for f in trajectory_list[i].trajectory[k*span:(k+1)*span]:    
                                position_0 = selection_0.positions
                                position_1 = selection_1.positions
                                ts_bond_vector = position_0 - position_1 
                                norm_vector = ts_bond_vector/np.linalg.norm(ts_bond_vector)
                                bond_vectors_normal[en3, :] = norm_vector
                                en3 +=1 
                            # print(calc_O2_from_traj_unit_vectors(bond_vectors_normal))
                            O2_bond[i*factor+k, nj] = qnmd.calc_O2_from_traj_unit_vectors(bond_vectors_normal)
                        
                O2_dict[aa_methyl] = {'resnums': aa_idx, 'O2_values': O2_bond.T}

    return O2_dict
    
    

def calc_O2_from_traj_unit_vectors(uvs):
    """
    Calculates the O2 order parameter from unit vectors of a trajectory.
    This function assumes uvs is an nx3 numpy matrix where each row represents a unit vector in 3D space.

    Parameters:
    - uvs (numpy.ndarray): an nx3 matrix where each row is a unit vector in 3D space.

    Returns:
    - float: the calculated O2 order parameter.
    """
    # Squaring individual components
    x_sq = uvs[:, 0]**2
    y_sq = uvs[:, 1]**2
    z_sq = uvs[:, 2]**2

    # Computing products of different components
    xy = uvs[:, 0] * uvs[:, 1]
    xz = uvs[:, 0] * uvs[:, 2]
    yz = uvs[:, 1] * uvs[:, 2]

    # Calculating the mean squared values and computing the final O2 formula
    return (3/2) * (x_sq.mean()**2 + y_sq.mean()**2 + z_sq.mean()**2 + 
                    2 * xy.mean()**2 + 2 * xz.mean()**2 + 2 * yz.mean()**2) - 0.5

def calc_O2_from_traj_unit_vectors_RW(uvs, weights):
    """
    Calculates the weighted O2 order parameter from unit vectors of a trajectory,
    weighted by a numpy array of weights.

    Parameters:
    - uvs (numpy.ndarray): an nx3 matrix where each row is a unit vector in 3D space.
    - weights (numpy.ndarray): an array of weights corresponding to each unit vector.

    Returns:
    - float: the calculated weighted O2 order parameter.
    """

    if uvs.shape[0] != weights.shape[0]:
        print('Size Mismatch between Unit Vectors and Weights. The number of unit vectors must match the number of weights.')
        return 0
    else:
        # Applying weights to products
        x_sq = weights * uvs[:, 0]**2
        y_sq = weights * uvs[:, 1]**2
        z_sq = weights * uvs[:, 2]**2
        xy = weights * uvs[:, 0] * uvs[:, 1]
        xz = weights * uvs[:, 0] * uvs[:, 2]
        yz = weights * uvs[:, 1] * uvs[:, 2]

        # Calculating the mean squared values and computing the final O2 formula
        return (3/2) * (x_sq.sum()**2 + y_sq.sum()**2 + z_sq.sum()**2 + 
                        2 * xy.sum()**2 + 2 * xz.sum()**2 + 2 * yz.sum()**2) - 0.5



def calc_phi_angles(trajectory_list: list, system_list: list):
    """
    Calculate phi dihedral angles for specified residues across multiple trajectories.
    
    Currently configured to calculate the phi dihedral angle (CB-CG-SD-CE) for methionine residues.
    The function computes dihedral angles for each residue in system_list across all frames
    in each trajectory.
    
    Parameters:
    -----------
    trajectory_list : list
        List of MDAnalysis Universe objects containing trajectory data.
    system_list : list
        List of residue numbers (resnums) to calculate angles for.
    
    Returns:
    --------
    angles : numpy.ndarray
        Array of shape (n_frames, n_traj, n_syst) containing phi dihedral angles in radians
        for each frame, trajectory, and system.
    """
    n_traj = len(trajectory_list)
    n_syst = len(system_list)
    n_frames = trajectory_list[0].trajectory.n_frames
    angles = np.zeros((n_frames, n_traj, n_syst))
    for i in range(n_traj):
        print(f'traj. #{i}')
        for j, s in enumerate(system_list):
            atoms_sel = trajectory_list[i].select_atoms(f"resnum {s} and (name CB or name CG or name SD or name CE)")
            phi_dihedral = dihedrals.Dihedral([atoms_sel])
            a=phi_dihedral.run().results.angles
            for k, aa in enumerate(a):
                angles[k, i, j]  = aa[0]
    return angles
    

def calc_theta_angles(trajectory_list: list, system_list: list):
    """
    Calculate theta bond angles for specified residues across multiple trajectories.
    
    Currently configured to calculate the theta bond angle (CG-SD-CE) for methionine residues.
    The function computes bond angles for each residue in system_list across all frames
    in each trajectory.
    
    Parameters:
    -----------
    trajectory_list : list
        List of MDAnalysis Universe objects containing trajectory data.
    system_list : list
        List of residue numbers (resnums) to calculate angles for.
    
    Returns:
    --------
    angles : numpy.ndarray
        Array of shape (n_frames, n_traj, n_syst) containing theta bond angles in radians
        for each frame, trajectory, and system.
    """
    n_traj = len(trajectory_list)
    n_syst = len(system_list)
    n_frames = trajectory_list[0].trajectory.n_frames
    angles = np.zeros((n_frames, n_traj, n_syst))
    for i in range(n_traj):
        print(f'traj. #{i}')
        for j, s in enumerate(system_list):
            atoms_sel = trajectory_list[i].select_atoms(f"resnum {s} and (name CE or name SD or name CG)")
            for k, ts in enumerate(trajectory_list[i].trajectory):
                theta_atoms = atoms_sel
                angles[k, i, j]  = theta_atoms.angle.value()
    return angles  

def thetaphi2uv(θ, φ):
    """
    Convert spherical coordinates (theta and phi angles) to a unit vector in Cartesian coordinates.
    
    Converts angles from degrees to a 3D unit vector (x, y, z) using spherical coordinate
    transformation. The formula uses (180 - θ) for the polar angle conversion.
    
    Parameters:
    -----------
    θ : float or numpy.ndarray
        Theta angle in degrees (polar angle, measured from z-axis).
    φ : float or numpy.ndarray
        Phi angle in degrees (azimuthal angle, measured from x-axis in xy-plane).
    
    Returns:
    --------
    tuple
        A tuple of three values (x, y, z) representing the unit vector components:
        - x: sin(180-θ) * cos(φ)
        - y: sin(180-θ) * sin(φ)
        - z: cos(180-θ)
    """
    fac = np.pi / 180
    return np.sin(fac*(180 - θ))*np.cos(fac*φ), np.sin(fac*(180 - θ))*np.sin(fac*φ), np.cos(fac*(180 - θ)) 


def calculate_Ct_Palmer(vecs):
    """
    Definition: < P2( v(t).v(t+dt) )  >
    (Rewritten) This proc assumes vecs to be of square dimensions ( nReplicates, nFrames, nResidues, 3).
    Operates a single einsum per delta-t timepoints to produce the P2(v(t).v(t+dt)) with dimensions ( nReplicates, nResidues )
    then produces the statistics from there according to Palmer's theory that trajectory be divide into N-replcates with a fixed memory time.
    Output Ct and dCt should take dimensions ( nResidues, nDeltas )
    """
    sh = vecs.shape
    #print "= = = Debug of calculate_Ct_Palmer confirming the dimensions of vecs:", sh
    if sh[1]<50:
        print("= = = WARNING: there are less than 50 frames per block of memory-time!", file=sys.stderr)

    if len(sh)!=4:
        # Not in the right form...
        #print >> sys.stderr, "= = = ERROR: The input vectors to calculate_Ct_Palmer is not of the expected 4-dimensional form! " % sh
        sys.exit(1)
    nReplicates = sh[0]
    nDeltas = sh[1] // 2
    nResidues = sh[2]
    Ct_ind = np.zeros( (nReplicates, nDeltas,nResidues) )
    Ct  = np.zeros( (nDeltas,nResidues) )
    dCt = np.zeros( (nDeltas,nResidues) )
    #bFirst=True
    for delta in range(1,1+nDeltas):
        nVals=sh[1]-delta
        # = = Create < vi.v'i > with dimensions (nRep, nFr, nRes, 3) -> (nRep, nFr, nRes) -> ( nRep, nRes ), then average across replicates with SEM.
        tmp = -0.5 + 1.5 * np.square( np.einsum( 'ijkl,ijkl->ijk', vecs[:,:-delta,...] ,vecs[:,delta:,...] ) ) #(nRep, nFr, nRes, 3) -> (nRep, nFr, nRes) over -delta to delta frames
        tmp  = np.einsum( 'ijk->ik', tmp ) / nVals # (nRep, nFr, nRes) -> ( nRep, nRes )
        Ct_ind[:, delta-1, :] = tmp
        Ct[delta-1]  = np.mean( tmp,axis=0 ) #( nRep, nRes ) -> ( nRes )
        dCt[delta-1] = np.std( tmp,axis=0 ) / ( np.sqrt(nReplicates) - 1.0 )
        #if bFirst:
        #    bFirst=False
        #    print tmp.shape, P2.shape
        #    print tmp[0,0,0], P2[0,0]
        #Ct[delta-1]  = np.mean( tmp,axis=(0,1) )
        #dCt[delta-1] = np.std( tmp,axis=(0,1) ) / ( np.sqrt(nReplicates*nVals) - 1.0 )

    #print("= = Bond %i Ct computed. Ct(%g) = %g , Ct(%g) = %g " % (i, dt[0], Ct_loc[0], dt[-1], Ct_loc[-1]))
    # Return with dimensions ( nDeltas, nResidues ) by default.
    return Ct_ind, Ct, dCt



def calc_unit_vectors(trajectory_list: list, resname: str='MET', atoms: list=['SD', 'CE']):
    """
    Calculate normalized bond vectors between two specified atoms for a given residue type.
    
    This function computes unit vectors representing the bond direction between two atoms
    (e.g., SD-CE for methionine) for all residues of the specified type across multiple
    trajectories. The bond vectors are normalized to unit length.
    
    Parameters:
    -----------
    trajectory_list : list
        List of MDAnalysis Universe objects containing trajectory data.
    resname : str, optional
        Residue name to select (default: 'MET' for methionine).
    atoms : list, optional
        List of two atom names to define the bond vector. The vector points from
        atoms[1] to atoms[0] (default: ['SD', 'CE']).
    
    Returns:
    --------
    norm_bond_vectors : numpy.ndarray
        Array of shape (n_frames, n_traj, n_systems, 3) containing normalized bond vectors.
        Each vector is a unit vector in 3D space representing the bond direction at each
        frame, trajectory, and residue instance.
    """
    n_traj = len(trajectory_list)
    n_frames = trajectory_list[0].trajectory.n_frames
    selection_string = f'resname {resname}'
    selection_systems = trajectory_list[0].select_atoms(selection_string)
    n_systems = selection_systems.n_residues
    system_idx = sorted(set(selection_systems.resids))
    print(f'Residue type: {resname} found at {system_idx} and there are {n_systems} of them.')
    #norm_bond_vectors = np.zeros((n_traj, n_frames, n_systems, 3))
    norm_bond_vectors = np.zeros((n_frames, n_traj, n_systems, 3))
    for n, u in enumerate(trajectory_list):
        selection_0 = u.select_atoms(f'resname {resname} and name {atoms[0]}')
        selection_1 = u.select_atoms(f'resname {resname} and name {atoms[1]}')
        for i, ts in enumerate(u.trajectory):
            # generate bond vectors
            
            position_0 = selection_0.positions
            position_1 = selection_1.positions
            ts_bond_vectors = position_0 - position_1 # num_systems selected x 3 (x, y, z)
        
            for j, a in enumerate(ts_bond_vectors):
                norm_bond_vectors[i, n, j, :] = a/np.linalg.norm(a)
        
            
    return norm_bond_vectors