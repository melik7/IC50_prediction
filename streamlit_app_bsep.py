######################
# Import libraries
######################

import numpy as np
from sklearn.svm import SVC
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw 
import altair as alt
from PIL import Image
import deepchem as dc
from mordred import Calculator, descriptors
from sklearn.decomposition import PCA, SparsePCA
import matplotlib.pyplot as plt 



X1 = pd.read_csv('Morgan_Warn_Mordred.csv')
# print(X1.head())
y_wm = X1['pIC50'] #< -np.log10(0.0000300)

y_2 = X1['active']
# y_2 = y_wm.copy()
# y_2[y_2<-np.log10(0.000030)] = 0
# y_2[y_2>=-np.log10(0.000030)] = 1

X_equi = X1.drop(['source', 'active', 'pIC50'], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(X_equi)

svm = SVC(C=10, gamma=0.0001, probability=True)
svm.fit(X_scale, y_2)


######################
# Page Title
######################

image = Image.open('3Dmoll.png')

st.image(image)

st.write("""
# Prediction model for BSEP transporter inhibition
This application predicts whether or not a molecule is a blocker of the BSEP transporter

***
""")


######################
# Input Text Box
######################

#st.sidebar.header('Enter DNA sequence')
st.header('Enter SMILES chain')

sequence_input = "c1cc(C(=O)O)c(OC(=O)C)cc1"

#sequence = st.sidebar.text_area("Sequence input", sequence_input, height=250)
sequence = st.text_area("Input", height=100)
# sequence = sequence.splitlines()
# sequence = sequence[1:] # Skips the sequence name (first line)
# sequence = ''.join(sequence) # Concatenates list to string
st.write("""
***
""")

print(sequence)

st.header('Molecular structure')

compound_smiles = 'c1cc(C(=O)O)c(OC(=O)C)cc1'
m = Chem.MolFromSmiles(sequence)
im=Draw.MolToImage(m)

st.image(im)

# calc = Calculator(descriptors, ignore_3D=True)
# mol = Chem.MolFromSmiles(sequence)
# query1 = calc(mol)
# query1 = np.array(query1)
# query1 = query1.reshape(1, query1.shape[0])
# query1 = query1.astype(float)

featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
query1 = featurizer.featurize(sequence)

query1_sc = scaler.transform(query1)

prediction = svm.predict(query1_sc)


label = np.array(['Non Blocker', 'Blocker'])
st.subheader("The molecule is a : ")
st.write(label[prediction])

st.subheader("The associated probabilities are : ")
st.write(svm.predict_proba(query1))

pca = PCA(n_components = 2)
D = pca.fit_transform(X_scale)

point = pca.transform(query1_sc)


# Plot of the individuals with a color by competition

fig, ax = plt.subplots()
ax.scatter(D[:,0][y_2==0],D[:,1][y_2==0],c='orange', label='Non blocker')
ax.scatter(D[:,0][y_2==1],D[:,1][y_2==1],c='blue', label='Blocker')
ax.scatter(point[:,0],point[:,1],c="r", label='Your molecule')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA from 1613 features')
ax.legend()

st.pyplot(fig)

# ## Prints the input DNA sequence
# st.header('INPUT (DNA Query)')
# sequence

# ## DNA nucleotide count
# st.header('OUTPUT (DNA Nucleotide Count)')

# ### 1. Print dictionary
# st.subheader('1. Print dictionary')
# def DNA_nucleotide_count(seq):
#   d = dict([
#             ('A',seq.count('A')),
#             ('T',seq.count('T')),
#             ('G',seq.count('G')),
#             ('C',seq.count('C'))
#             ])
#   return d

# X = DNA_nucleotide_count(sequence)

# #X_label = list(X)
# #X_values = list(X.values())

# X

# ### 2. Print text
# st.subheader('2. Print text')
# st.write('There are  ' + str(X['A']) + ' adenine (A)')
# st.write('There are  ' + str(X['T']) + ' thymine (T)')
# st.write('There are  ' + str(X['G']) + ' guanine (G)')
# st.write('There are  ' + str(X['C']) + ' cytosine (C)')

# ### 3. Display DataFrame
# st.subheader('3. Display DataFrame')
# df = pd.DataFrame.from_dict(X, orient='index')
# df = df.rename({0: 'count'}, axis='columns')
# df.reset_index(inplace=True)
# df = df.rename(columns = {'index':'nucleotide'})
# st.write(df)

# ### 4. Display Bar Chart using Altair
# st.subheader('4. Display Bar chart')
# p = alt.Chart(df).mark_bar().encode(
#     x='nucleotide',
#     y='count'
# )
# p = p.properties(
#     width=alt.Step(80)  # controls width of bar.
# )
# st.write(p)