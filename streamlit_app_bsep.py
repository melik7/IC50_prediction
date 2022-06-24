######################
# Import libraries
######################

import numpy as np
from sklearn.svm import SVC, SVR
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



def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():


    X1 = pd.read_csv('Morgan_Warn_Mordred.csv')
    # print(X1.head())
    y_wm = X1['pIC50'] #< -np.log10(0.0000300)
    
    
    y_modif = y_wm.copy()
    y_modif[X1['source'] == 'Morgan'] = y_modif[X1['source'] == 'Morgan'] + 0.35
    
    y_2 = y_modif.copy()
    y_2[y_2<-np.log10(0.000030)] = 0
    y_2[y_2>=-np.log10(0.000030)] = 1
    
    y_3 = y_modif.copy()
    y_3[y_3<-np.log10(0.0000600)] = 0
    y_3[y_3>=-np.log10(0.0000300)] = 2
    y_3[(y_3<-np.log10(0.0000300)) & (y_3>=-np.log10(0.0000600))] = 1 
    
    # y_2 = X1['active']
    
    X_equi = X1.drop(['source', 'active', 'pIC50'], axis=1)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X_equi)
    
    svm = SVC(C=10, gamma=0.0001, probability=True)
    svm.fit(X_scale, y_2)
    
    svm_reg = SVR(C=10, gamma=0.0001)
    svm_reg.fit(X_scale, y_modif)
    
    ######################
    # Page Title
    ######################
    
    image = Image.open('3Dmoll.png')
    
    st.image(image)
    
    st.write("""
    # Prediction model for BSEP transporter inhibition
    This application predicts whether or not a molecule is a blocker of the BSEP transporter with a threshold of 30 µmol/l
    
    ***
    """)
    
    
    ######################
    # Input Text Box
    ######################
    
    #st.sidebar.header('Enter DNA sequence')
    st.header('Enter SMILES chain')
    
    # sequence_input = "c1cc(C(=O)O)c(OC(=O)C)cc1"
    
    #sequence = st.sidebar.text_area("Sequence input", sequence_input, height=250)
    sequence = st.text_area("Input", height=100)
    # sequence = sequence.splitlines()
    # sequence = sequence[1:] # Skips the sequence name (first line)
    # sequence = ''.join(sequence) # Concatenates list to string
    st.write("""
    ***
    """)
    
    
    
    # button = st.button('Enter free energy')
    # if button:
    #     number = st.slider('Free energy', 0.0, -10.0, key='slider')
    
    
    # compound_smiles = 'c1cc(C(=O)O)c(OC(=O)C)cc1'
    
    
    
    
    if sequence != '':
        st.subheader('Molecular structure')
        m = Chem.MolFromSmiles(sequence)
        im=Draw.MolToImage(m)
        st.image(im)
    
        # calc = Calculator(descriptors, ignore_3D=True)
        # mol = Chem.MolFromSmiles(sequence)
        # query1 = calc(mol)
        # query1 = np.array(query1)
        # query1 = query1.reshape(1, query1.shape[0])
        # query1 = query1.astype(float)
        
        featurizer = dc.feat.MordredDescriptors(ignore_3D=False)
        query2 = featurizer.featurize(sequence)
        
        feats = query2[:,[1818, 1766, 1304, 1305, 1764, 1769]]
        tab_df = pd.DataFrame(feats)
        tab_df.columns = ["MW","SlogP","AcceptorH","DonorH","Rotatable_bond", 'TopoPSA']
        st.subheader('Estimation of the main parameters')
        st.write(tab_df)
        
        featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
        query1 = featurizer.featurize(sequence)
        
        query1_sc = scaler.transform(query1)
        
        prediction = svm.predict(query1_sc)
        
        st.write("""
    ***
    """)
        
        
        label = np.array(['Non Blocker', 'Blocker'])
        answer1 = label[int(prediction)]
        st.header("The molecule is predicted as a : ")
        st.write(answer1)
        
        
        answer2 = svm.predict_proba(query1)
        st.header("The associated probabilities are : ")
        st.write(answer2)
        
        st.header("The predicted value of IC 50 (µmol/l) is : ")
        pic50 = svm_reg.predict(query1_sc)[0]
        ic50 = round(10**-pic50*10**6, 2)
        st.write(str(ic50), 'µM')
        
        pca = PCA(n_components = 2)
        D = pca.fit_transform(X_scale)
        
        point = pca.transform(query1_sc)
        
        st.write("""
    ***
    """)
        
        
        # Plot of the individuals with a color by competition
        
        # fig, ax = plt.subplots()
        # ax.scatter(D[:,0][y_2==0],D[:,1][y_2==0],c='orange', label='Non blocker')
        # ax.scatter(D[:,0][y_2==1],D[:,1][y_2==1],c='blue', label='Blocker')
        # ax.scatter(point[:,0],point[:,1],c="r", label='Your molecule')
        # ax.set_xlabel('PC1')
        # ax.set_ylabel('PC2')
        # ax.set_title('PCA from 1613 features')
        # ax.legend()
        
       
        
        fig, ax = plt.subplots()
        ax.scatter(D[:,0][y_2==0],D[:,1][y_2==0],c='orange', label='Non blocker')
        ax.scatter(D[:,0][y_2==1],D[:,1][y_2==1],c='deepskyblue', label='Blocker')
        ax.scatter(point[:,0],point[:,1],c="r", label='Your molecule')
        ax.scatter(10.95365287,  2.37109774, marker='+', c='black', label='Validation compound')
        ax.text(11.95365287,  3.37109774, 'Ketoconazole', c='black')
        
        ax.scatter(46.97309365, -1.46460239, marker='+', c='black')
        ax.text(47.97309365, -0.46460239, 'Rifampicin', c='black')
        
        ax.scatter(-20.05171732,   4.52928818, marker='+', c='black')
        ax.text(-19.05171732,   5.52928818, 'Caffeine', c='black')
        
        ax.scatter(81.65933912, -16.0876484, marker='+', c='black')
        ax.text(82.65933912, -15.0876484, 'Cyclosporine A', c='black')
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA from 1613 features')
        ax.legend()
        
        st.subheader('Data visualization')
        st.pyplot(fig)
        
        
        # plt.scatter(D[:,0][y_2==0],D[:,1][y_2==0],c='orange', label='Non blocker')
        # plt.scatter(D[:,0][y_2==1],D[:,1][y_2==1],c='blue', label='Blocker')
        # plt.scatter(point[:,0],point[:,1],c="r", label='Your molecule')
        # plt.xlabel('PC1')
        # plt.ylabel('PC2')
        # plt.title('PCA from 1613 features')
        # plt.legend()
        # plt.show()
        
        # text_contents = '''
        # Foo, Bar
        # 123, 456
        # 789, 000
        # '''
        
        df = pd.DataFrame(np.zeros((2,2)))
        name = ["MW","SlogP","AcceptorH","DonorH","Rotatable_bond", "TopoPSA"]
        feature_val = ''
        for i,j in zip(feats[0], name):
            feature_val += j + ' : ' + str(round(i,2)) + '\n'
        
        output = '*** SMILE query : ***' + '\n'+ sequence + '\n\n\n'+ '*** Estimation of physico-chemical parameters : ***' + '\n' + \
        feature_val + '\n' + '*** Prediction model results : ***' + '\n' + 'Molecule predicted as a : ' + answer1 + '\n' + 'Probability associated : '\
        + str(round(answer2[0][0]*100,2)) + ' %' + '\n' + 'Value predicted : ' + str(ic50) + ' µM'
            
        # Different ways to use the API
        
        # st.download_button('Download CSV', text_contents, 'text/csv.txt')
        
        # st.download_button('Download CSV', text_contents)  # Defaults to 'text/plain'
        st.download_button('Download the report', output, file_name='Report.txt')  # Defaults to 'text/plain'
    
    
        
    
