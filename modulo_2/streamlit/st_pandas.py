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


    st.title("Modulo 2 - Streamlit - AceleraDev")
    st.image("codenation_logo.png")
    file = st.file_uploader("Upload your file", )
    #file  = "titanic_train.csv"
    if file is not None:
        slider = st.slider("Registros de 1 a 100", 1,100)
        df = pd.read_csv(file)
        st.dataframe(df.head(slider))
        #st.table(df.head(slider))
        st.write(df.columns)
        #st.dataframe(df.groupby('Sex').agg({'Age':'mean','Survived':'mean'}))
        st.table(df.groupby('Sex').agg({'Age':'mean','Survived':'mean'}))

#    st.markdown("Botão")
#    bt = st.button("Botão")
#    if bt:
#        st.markdown("Clicado")
#
#    st.markdown("Checkbox")   
#    checkbox = st.checkbox("Checkbox")
#    if checkbox:
#        st.markdown("Clicado")
#
#    st.markdown("Radio")    
#    radio = st.radio("Escolha as opções",("a", "b", "c"))
#    if radio == "a":
#        st.markdown("a")
#    if radio == "b":
#        st.markdown("b")
#    if radio == "c":
#        st.markdown("c")
#
#    st.markdown("Selectbox")
#    select = st.selectbox("Choose option", ("a", "b", "c"))
#    if select == "a":
#        st.markdown("a")
#    if select == "b":
#        st.markdown("b")
#    if select == "c":
#        st.markdown("c")
#
#    multi = st.multiselect("Choose", ("a", "b", "c"))
#    if multi == "a":
#        st.markdown("a")
#    if multi == "b":
#        st.markdown("b")
#    if multi == "c":
#        st.markdown("c")
#
#    st.markdown("File uploaded")
#    file = st.file_uploader("Choose your file",type ="csv")
#    if file is not None:
#        st.markdown("Não está vazio")



if __name__ == '__main__':
    main()