

import streamlit as st
import numpy as np
import pandas as pd

books=pd.read_csv(r"C:/Users/Athira A R/Desktop/Book Recommendation -MLP/Books.csv",low_memory=False)
users=pd.read_csv(r"C:/Users/Athira A R/Desktop/Book Recommendation -MLP/Users.csv")
ratings=pd.read_csv(r"C:/Users/Athira A R/Desktop/Book Recommendation -MLP/Ratings.csv")


ratings_with_name=ratings.merge(books,on='ISBN')

num_rating_df=ratings_with_name.groupby('Book-Title').count()['Book-Rating']  #group by Book_Title and count the no of ratings
num_rating_df=num_rating_df.reset_index()                                                   #Reset the index to convert the result into a DataFrame
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)                    #Rename the 'Book-Rating' column to 'num_ratings'


avg_rating_df=ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating']   #group the dataframe by Book-Title and find mean of all numeric cols,[Book_rating] > will select only the Book-rating col from the result.
avg_rating_df=avg_rating_df.reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_ratings'},inplace=True)

popular_df=num_rating_df.merge(avg_rating_df,on='Book-Title')


top_50=popular_df[popular_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False).head(50)

final_top_50=top_50.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_ratings']]


url_value = final_top_50['Image-URL-M'].iloc[0]



x=ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
knowledgeble_users=x[x].index

filtered_rating=ratings_with_name[ratings_with_name['User-ID'].isin(knowledgeble_users)]

#filtered_rating will be a new DataFrame that contains only the rows corresponding to users who have rated more than 200 books, based on the knowledgeble_users list.

y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index                  #y[y].index: This line filters the books for which the count is greater than or equal to 50 and extracts their indices.

final_ratings=filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]    #final_ratings will be a new DataFrame that contains only the rows corresponding to books that have received 50 or more ratings from the knowledgeable users, based on the famous_books list.

pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')

#created a pivot table from a DataFrame'final_ratings', where 'Book-Title' served as the row index, 'User-ID' as the column index, and 'Book-Rating' as the values. This allows for easy exploration of how different users have rated various books

pt.fillna(0,inplace=True)





from sklearn.metrics.pairwise import cosine_similarity

similarity_scores=cosine_similarity(pt)



def recommend(book_name):
    try:
        index = np.where(pt.index == book_name)[0]
        if len(index) == 0:
            return None  # Book not found
        index = index[0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]

        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

            data.append(item)
        return data
    except Exception as e:
        print("Error:", e)
        return None




import pickle
pickle.dump(final_top_50,open('final_top_50.pkl','wb'))

books.drop_duplicates('Book-Title')

# Save pt, books, and similarity_scores to separate files

pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))

def main():
    st.title('Book Recommendation App')

    # Input field for entering book title
    book_title = st.text_input("Enter the book title:")
    recommended_books = recommend(book_title)
    if recommended_books is not None:
        st.subheader("Recommended Books:")
        for book in recommended_books:
            st.write(f"**Name:** {book[0]}")
            st.write(f"**Author:** {book[1]}")
            st.image(book[2], caption='Book Cover', use_column_width=True)
            
            
        
            
            
if __name__ == "__main__":
    
    pt = pickle.load(open('pt.pkl', 'rb'))  # Define pt
    similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))  # Define similarity_scores
    books = pickle.load(open('books.pkl', 'rb')) # Define books DataFrame
    main()
