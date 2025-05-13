from Bio.PDB import DSSP 

def get_RSA(model, path) :
    dssp_ret = DSSP(model, path)
    AA = [i[1] for i in dssp_ret.property_list]
    RSA = [i[3] for i in dssp_ret.property_list]
    
    pdb_seq = ''.join(AA)
    
    return pdb_seq,RSA