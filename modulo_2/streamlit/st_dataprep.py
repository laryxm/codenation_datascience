import streamlit as st
import pandas as pd

def main():

#    st.title("Hello!")
#    st.header("This is a header!")
#    st.subheader("This is a subheader!")
#    st.text("This is a text!")
#    #st.image("logo.png")
#    st.subheader("Audio")
#    #st.audio("record.wave")
#    st.subheader("Vídeo")
#    #st.video("sentiment_motion.mov")

    # st.markdown("File uploaded")

    #st.title("Data Preparation - Codenation DS")
    st.image("codenation_logo.png")


    file = st.file_uploader("Choose your file",type ="csv")
    if file is not None:
        st.markdown("Não está vazio")

   
    file  = "titanic_train.csv"

    if file is not None: 
        slider = st.slider("Registros de 5 a 100", 5,100)
        df = pd.read_csv(file)
        st.dataframe(df.head(slider))
        st.write("Dataset shape:" , df.shape)

        # Describe the dataset
        st.write(df.describe())

        # Select the dataset keys
        multiselect_keys = st.multiselect("Choose the keys", (df.columns))
        st.write("We have " , df[multiselect_keys].drop_duplicates().shape[0], " distinct key values.")

        # Select the target
        select_target = st.selectbox("Select the target", (df.columns), index  = 1)    
        
        # List the columns types
        int_col_list = df.dtypes[df.dtypes == "int64"].index
        float_col_list = df.dtypes[df.dtypes == "float64"].index
        object_col_list = df.dtypes[df.dtypes == "object"].index

        # Show the target  ratio
        if ("int" in str(df.dtypes.loc[select_target])):
            st.write(df[select_target].value_counts(normalize = True))
        elif ("object" in str(df.dtypes.loc[select_target])):
            st.write(df[select_target].value_counts(normalize = True))
        elif ("float" in str(df.dtypes.loc[select_target])):
            st.write(df[select_target].mean())
        
        # Show the total of nulls in the dataset 
        st.markdown("nan values")
        #st.table(df.head(slider))
        st.dataframe(df.isna().sum(), width=500, height=500)
        #st.dataframe(df.groupby('Sex').agg({'Age':'mean','Survived':'mean'}))
        st.dataframe(df.groupby('Sex').agg({'Age':'mean','Survived':'mean'}))


    st.title("Hello!")
    
    st.markdown("Botão")
    bt = st.button("Botão")
    if bt:
        st.markdown("Clicado")

    st.markdown("Checkbox")   
    checkbox = st.checkbox("Checkbox")
    if checkbox:
        st.markdown("Clicado")

    st.markdown("Radio")    
    radio = st.radio("Escolha as opções",("a", "b", "c"))
    if radio == "a":
        st.markdown("a")
    if radio == "b":
        st.markdown("b")
    if radio == "c":
        st.markdown("c")

    st.markdown("Selectbox")
    select = st.selectbox("Choose option", ("a", "b", "c"))
    if select == "a":
        st.markdown("a")
    if select == "b":
        st.markdown("b")
    if select == "c":
        st.markdown("c")

    multi = st.multiselect("Choose", ("a", "b", "c"))
    if multi == "a":
        st.markdown("a")
    if multi == "b":
        st.markdown("b")
    if multi == "c":
        st.markdown("c")





if __name__ == '__main__':
    main()