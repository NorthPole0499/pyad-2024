def personal_recommedations(books, ratings, lin_model, svd_model):
  ratings_zero = ratings[ratings["Book-Rating"] == 0].groupby("User-ID")["Book-Rating"].count()
  current_user = ratings_zero.idxmax()

  zero_books = ratings[(ratings["User-ID"] == current_user) & (ratings["Book-Rating"] == 0)]["ISBN"].unique()

  test_svd = []
  for elem in zero_books:
    test_svd.append((current_user, elem, 0))
  predictions_svd = svd_model.test(test_svd)

  eight_books = []
  for element in predictions_svd:
    if element.est >= 8:
      eight_books.append(element.iid)
  
  books_data = books[books["ISBN"].isin(eight_books)].copy()

  X_test = books_data[["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]]
  prediction_lin = lin_model.predict(X_test)

  books_data["Predicted-Rating"] = prediction_lin
  books_data.sort_values("Predicted-Rating", ascending=False, inplace=True)

  recommendation = books_data[["ISBN", "Book-Title", "Book-Author", "Predicted-Rating"]].head(10)

  print("Топ-10 книг")
  print(recommendation.to_string(index=False))

  """
  Полученная рекомендация
  -----------------------
  ISBN                                                        Book-Title         Book-Author  Predicted-Rating
0439064872                               Harry Potter Chamber Secrets Book 2       J. K. Rowling          8.459338
0451194861                                    Wizard Glass Dark Tower Book 4        Stephen King          8.422258
059035342X             Harry Potter Sorcerer 's Stone Harry Potter Paperback       J. K. Rowling          8.210303
044098761X                                       Wind Door Laurel Leaf Books   Madeleine L'Engle          8.179818
0694003611                                         Goodnight Moon Board Book Margaret Wise Brown          8.112803
0345313860                         Vampire Lestat Vampire Chronicles Book II           ANNE RICE          8.092775
0345339681                              Hobbit Enchanting Prelude Lord Rings      J.R.R. TOLKIEN          8.082581
0064409422               Lion Witch Wardrobe Full-Color Collector 's Edition         C. S. Lewis          8.023103
039480029X                                       Hop Pop Read Beginner Books           Dr. Seuss          7.947755
0064401707 Scary Stories Tell Dark Collected American Folklore Scary Stories      Alvin Schwartz          7.934798
  """
