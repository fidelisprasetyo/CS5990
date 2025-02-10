# -------------------------------------------------------------------------
# AUTHOR: Fidelis Prasetyo
# FILENAME: similarity.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 1 day
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append(row)
         #print(row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.

termsIndex = {}
idx = 0

for doc in documents:
   words = doc[1].split()
   for word in words:
      if word not in termsIndex:
        termsIndex[word] = idx
        idx += 1

docTermMatrix = []
for doc in documents:
   doc_matrix = [0] * len(termsIndex)
   words = doc[1].split()
   for word in words:
      doc_matrix[termsIndex[word]] += 1
   docTermMatrix.append(doc_matrix)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors

similarity = cosine_similarity(docTermMatrix, docTermMatrix)

# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x

max_similarity = -1
doc_idx = (-1,-1)
matrix_size = len(similarity)

for i in range(matrix_size):
   for j in range(i+1, matrix_size):
      if similarity[i][j] > max_similarity:
         max_similarity = similarity[i][j]
         doc_idx = (i,j)

print("The most similar documents are document " + str(doc_idx[0]+1) + " and document " + str(doc_idx[1]+1) + " with cosine similarity = " + str(max_similarity)) 
