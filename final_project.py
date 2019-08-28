from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
import pubchempy as pcp
import pandas as pd
import pickle
#example testing: coing from inchi to testable features
'''
m = Chem.MolFromSmiles('Cc1ccccc1')
x=Chem.MolToMolBlock(m)
#print(x)
test = pcp.get_compounds('InChI=1S/C15H14O6/c16-8-4-11(18)9-6-13(20)15(21-14(9)5-8)7-1-2-10(17)12(19)3-7/h1-5,13,15-20H,6H2','inchi')
df4 = pcp.compounds_to_frame(test, properties=['cactvs_fingerprint','isomeric_smiles', 'xlogp', 'rotatable_bond_count','charge','complexity','exact_mass','fingerprint'])
'''

#loading initial data
dataframe = pd.read_csv('retention_time_testdata.csv')
newdf = pd.DataFrame()
for index, row in dataframe.iterrows():
    if index == 0:
        inchi = row['InChI']
        cmpd = pcp.get_compounds(inchi,'inchi')
        props = cmpd[0].to_dict(properties=['cactvs_fingerprint','isomeric_smiles', 'xlogp', 'rotatable_bond_count','charge','complexity','exact_mass','fingerprint'])
        props['RT'] = row['RT']
        newdf=pd.DataFrame(props,index=[index])
    else:
        inchi = row['InChI']
        try:
            cmpd = pcp.get_compounds(inchi,'inchi')
        except:
            print('line bypassed')
            pass
        try:
            props = cmpd[0].to_dict(properties=['cactvs_fingerprint','isomeric_smiles', 'xlogp', 'rotatable_bond_count','charge','complexity','exact_mass','fingerprint'])
        except:
            print('line bypassed')
            pass
        props['RT'] = row['RT']
        newdf=newdf.append(props,ignore_index=True)
        print('on index ' + str(index) + ' of ' + str(len(dataframe)))
newdf.to_pickle('my_df.pickle')