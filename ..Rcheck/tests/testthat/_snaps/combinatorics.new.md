# combinations with repetition algorithm works

    Code
      combinations_with_repetition(5, 3)
    Output
            [,1] [,2] [,3]
       [1,]    1    1    1
       [2,]    1    1    2
       [3,]    1    1    3
       [4,]    1    1    4
       [5,]    1    1    5
       [6,]    1    2    2
       [7,]    1    2    3
       [8,]    1    2    4
       [9,]    1    2    5
      [10,]    1    3    3
      [11,]    1    3    4
      [12,]    1    3    5
      [13,]    1    4    4
      [14,]    1    4    5
      [15,]    1    5    5
      [16,]    2    2    2
      [17,]    2    2    3
      [18,]    2    2    4
      [19,]    2    2    5
      [20,]    2    3    3
      [21,]    2    3    4
      [22,]    2    3    5
      [23,]    2    4    4
      [24,]    2    4    5
      [25,]    2    5    5
      [26,]    3    3    3
      [27,]    3    3    4
      [28,]    3    3    5
      [29,]    3    4    4
      [30,]    3    4    5
      [31,]    3    5    5
      [32,]    4    4    4
      [33,]    4    4    5
      [34,]    4    5    5
      [35,]    5    5    5

---

    Code
      combinations_with_repetition(3, 5)
    Output
            [,1] [,2] [,3] [,4] [,5]
       [1,]    1    1    1    1    1
       [2,]    1    1    1    1    2
       [3,]    1    1    1    1    3
       [4,]    1    1    1    2    2
       [5,]    1    1    1    2    3
       [6,]    1    1    1    3    3
       [7,]    1    1    2    2    2
       [8,]    1    1    2    2    3
       [9,]    1    1    2    3    3
      [10,]    1    1    3    3    3
      [11,]    1    2    2    2    2
      [12,]    1    2    2    2    3
      [13,]    1    2    2    3    3
      [14,]    1    2    3    3    3
      [15,]    1    3    3    3    3
      [16,]    2    2    2    2    2
      [17,]    2    2    2    2    3
      [18,]    2    2    2    3    3
      [19,]    2    2    3    3    3
      [20,]    2    3    3    3    3
      [21,]    3    3    3    3    3

# multiset partitions are correctly generated

    Code
      generate_partitions(5, 3)
    Output
      [[1]]
      [[1]]$multiset
      [1] 1

      [[1]]$partitions
      [[1]]$partitions[[1]]
      [[1]]$partitions[[1]][[1]]
      [1] 1




      [[2]]
      [[2]]$multiset
      [1] 1 1

      [[2]]$partitions
      [[2]]$partitions[[1]]
      [[2]]$partitions[[1]][[1]]
      [1] 1 1


      [[2]]$partitions[[2]]
      [[2]]$partitions[[2]][[1]]
      [1] 1

      [[2]]$partitions[[2]][[2]]
      [1] 1




      [[3]]
      [[3]]$multiset
      [1] 1 2

      [[3]]$partitions
      [[3]]$partitions[[1]]
      [[3]]$partitions[[1]][[1]]
      [1] 1 2


      [[3]]$partitions[[2]]
      [[3]]$partitions[[2]][[1]]
      [1] 1

      [[3]]$partitions[[2]][[2]]
      [1] 2




      [[4]]
      [[4]]$multiset
      [1] 1 1 1

      [[4]]$partitions
      [[4]]$partitions[[1]]
      [[4]]$partitions[[1]][[1]]
      [1] 1 1 1


      [[4]]$partitions[[2]]
      [[4]]$partitions[[2]][[1]]
      [1] 1 1

      [[4]]$partitions[[2]][[2]]
      [1] 1


      [[4]]$partitions[[3]]
      [[4]]$partitions[[3]][[1]]
      [1] 1

      [[4]]$partitions[[3]][[2]]
      [1] 1

      [[4]]$partitions[[3]][[3]]
      [1] 1




      [[5]]
      [[5]]$multiset
      [1] 1 1 2

      [[5]]$partitions
      [[5]]$partitions[[1]]
      [[5]]$partitions[[1]][[1]]
      [1] 1 1 2


      [[5]]$partitions[[2]]
      [[5]]$partitions[[2]][[1]]
      [1] 1 1

      [[5]]$partitions[[2]][[2]]
      [1] 2


      [[5]]$partitions[[3]]
      [[5]]$partitions[[3]][[1]]
      [1] 1 2

      [[5]]$partitions[[3]][[2]]
      [1] 1


      [[5]]$partitions[[4]]
      [[5]]$partitions[[4]][[1]]
      [1] 1

      [[5]]$partitions[[4]][[2]]
      [1] 1

      [[5]]$partitions[[4]][[3]]
      [1] 2




      [[6]]
      [[6]]$multiset
      [1] 1 2 3

      [[6]]$partitions
      [[6]]$partitions[[1]]
      [[6]]$partitions[[1]][[1]]
      [1] 1 2 3


      [[6]]$partitions[[2]]
      [[6]]$partitions[[2]][[1]]
      [1] 1 2

      [[6]]$partitions[[2]][[2]]
      [1] 3


      [[6]]$partitions[[3]]
      [[6]]$partitions[[3]][[1]]
      [1] 1 3

      [[6]]$partitions[[3]][[2]]
      [1] 2


      [[6]]$partitions[[4]]
      [[6]]$partitions[[4]][[1]]
      [1] 1

      [[6]]$partitions[[4]][[2]]
      [1] 2 3


      [[6]]$partitions[[5]]
      [[6]]$partitions[[5]][[1]]
      [1] 1

      [[6]]$partitions[[5]][[2]]
      [1] 2

      [[6]]$partitions[[5]][[3]]
      [1] 3





---

    Code
      generate_partitions(3, 5)
    Output
      [[1]]
      [[1]]$multiset
      [1] 1

      [[1]]$partitions
      [[1]]$partitions[[1]]
      [[1]]$partitions[[1]][[1]]
      [1] 1




      [[2]]
      [[2]]$multiset
      [1] 1 1

      [[2]]$partitions
      [[2]]$partitions[[1]]
      [[2]]$partitions[[1]][[1]]
      [1] 1 1


      [[2]]$partitions[[2]]
      [[2]]$partitions[[2]][[1]]
      [1] 1

      [[2]]$partitions[[2]][[2]]
      [1] 1




      [[3]]
      [[3]]$multiset
      [1] 1 2

      [[3]]$partitions
      [[3]]$partitions[[1]]
      [[3]]$partitions[[1]][[1]]
      [1] 1 2


      [[3]]$partitions[[2]]
      [[3]]$partitions[[2]][[1]]
      [1] 1

      [[3]]$partitions[[2]][[2]]
      [1] 2




      [[4]]
      [[4]]$multiset
      [1] 1 1 1

      [[4]]$partitions
      [[4]]$partitions[[1]]
      [[4]]$partitions[[1]][[1]]
      [1] 1 1 1


      [[4]]$partitions[[2]]
      [[4]]$partitions[[2]][[1]]
      [1] 1 1

      [[4]]$partitions[[2]][[2]]
      [1] 1


      [[4]]$partitions[[3]]
      [[4]]$partitions[[3]][[1]]
      [1] 1

      [[4]]$partitions[[3]][[2]]
      [1] 1

      [[4]]$partitions[[3]][[3]]
      [1] 1




      [[5]]
      [[5]]$multiset
      [1] 1 1 2

      [[5]]$partitions
      [[5]]$partitions[[1]]
      [[5]]$partitions[[1]][[1]]
      [1] 1 1 2


      [[5]]$partitions[[2]]
      [[5]]$partitions[[2]][[1]]
      [1] 1 1

      [[5]]$partitions[[2]][[2]]
      [1] 2


      [[5]]$partitions[[3]]
      [[5]]$partitions[[3]][[1]]
      [1] 1 2

      [[5]]$partitions[[3]][[2]]
      [1] 1


      [[5]]$partitions[[4]]
      [[5]]$partitions[[4]][[1]]
      [1] 1

      [[5]]$partitions[[4]][[2]]
      [1] 1

      [[5]]$partitions[[4]][[3]]
      [1] 2




      [[6]]
      [[6]]$multiset
      [1] 1 2 3

      [[6]]$partitions
      [[6]]$partitions[[1]]
      [[6]]$partitions[[1]][[1]]
      [1] 1 2 3


      [[6]]$partitions[[2]]
      [[6]]$partitions[[2]][[1]]
      [1] 1 2

      [[6]]$partitions[[2]][[2]]
      [1] 3


      [[6]]$partitions[[3]]
      [[6]]$partitions[[3]][[1]]
      [1] 1 3

      [[6]]$partitions[[3]][[2]]
      [1] 2


      [[6]]$partitions[[4]]
      [[6]]$partitions[[4]][[1]]
      [1] 1

      [[6]]$partitions[[4]][[2]]
      [1] 2 3


      [[6]]$partitions[[5]]
      [[6]]$partitions[[5]][[1]]
      [1] 1

      [[6]]$partitions[[5]][[2]]
      [1] 2

      [[6]]$partitions[[5]][[3]]
      [1] 3




      [[7]]
      [[7]]$multiset
      [1] 1 1 1 1

      [[7]]$partitions
      [[7]]$partitions[[1]]
      [[7]]$partitions[[1]][[1]]
      [1] 1 1 1 1


      [[7]]$partitions[[2]]
      [[7]]$partitions[[2]][[1]]
      [1] 1 1 1

      [[7]]$partitions[[2]][[2]]
      [1] 1


      [[7]]$partitions[[3]]
      [[7]]$partitions[[3]][[1]]
      [1] 1 1

      [[7]]$partitions[[3]][[2]]
      [1] 1 1


      [[7]]$partitions[[4]]
      [[7]]$partitions[[4]][[1]]
      [1] 1 1

      [[7]]$partitions[[4]][[2]]
      [1] 1

      [[7]]$partitions[[4]][[3]]
      [1] 1


      [[7]]$partitions[[5]]
      [[7]]$partitions[[5]][[1]]
      [1] 1

      [[7]]$partitions[[5]][[2]]
      [1] 1

      [[7]]$partitions[[5]][[3]]
      [1] 1

      [[7]]$partitions[[5]][[4]]
      [1] 1




      [[8]]
      [[8]]$multiset
      [1] 1 1 1 2

      [[8]]$partitions
      [[8]]$partitions[[1]]
      [[8]]$partitions[[1]][[1]]
      [1] 1 1 1 2


      [[8]]$partitions[[2]]
      [[8]]$partitions[[2]][[1]]
      [1] 1 1 1

      [[8]]$partitions[[2]][[2]]
      [1] 2


      [[8]]$partitions[[3]]
      [[8]]$partitions[[3]][[1]]
      [1] 1 1 2

      [[8]]$partitions[[3]][[2]]
      [1] 1


      [[8]]$partitions[[4]]
      [[8]]$partitions[[4]][[1]]
      [1] 1 1

      [[8]]$partitions[[4]][[2]]
      [1] 1 2


      [[8]]$partitions[[5]]
      [[8]]$partitions[[5]][[1]]
      [1] 1 1

      [[8]]$partitions[[5]][[2]]
      [1] 1

      [[8]]$partitions[[5]][[3]]
      [1] 2


      [[8]]$partitions[[6]]
      [[8]]$partitions[[6]][[1]]
      [1] 1 2

      [[8]]$partitions[[6]][[2]]
      [1] 1

      [[8]]$partitions[[6]][[3]]
      [1] 1


      [[8]]$partitions[[7]]
      [[8]]$partitions[[7]][[1]]
      [1] 1

      [[8]]$partitions[[7]][[2]]
      [1] 1

      [[8]]$partitions[[7]][[3]]
      [1] 1

      [[8]]$partitions[[7]][[4]]
      [1] 2




      [[9]]
      [[9]]$multiset
      [1] 1 1 2 2

      [[9]]$partitions
      [[9]]$partitions[[1]]
      [[9]]$partitions[[1]][[1]]
      [1] 1 1 2 2


      [[9]]$partitions[[2]]
      [[9]]$partitions[[2]][[1]]
      [1] 1 1 2

      [[9]]$partitions[[2]][[2]]
      [1] 2


      [[9]]$partitions[[3]]
      [[9]]$partitions[[3]][[1]]
      [1] 1 1

      [[9]]$partitions[[3]][[2]]
      [1] 2 2


      [[9]]$partitions[[4]]
      [[9]]$partitions[[4]][[1]]
      [1] 1 1

      [[9]]$partitions[[4]][[2]]
      [1] 2

      [[9]]$partitions[[4]][[3]]
      [1] 2


      [[9]]$partitions[[5]]
      [[9]]$partitions[[5]][[1]]
      [1] 1 2 2

      [[9]]$partitions[[5]][[2]]
      [1] 1


      [[9]]$partitions[[6]]
      [[9]]$partitions[[6]][[1]]
      [1] 1 2

      [[9]]$partitions[[6]][[2]]
      [1] 1 2


      [[9]]$partitions[[7]]
      [[9]]$partitions[[7]][[1]]
      [1] 1 2

      [[9]]$partitions[[7]][[2]]
      [1] 1

      [[9]]$partitions[[7]][[3]]
      [1] 2


      [[9]]$partitions[[8]]
      [[9]]$partitions[[8]][[1]]
      [1] 1

      [[9]]$partitions[[8]][[2]]
      [1] 1

      [[9]]$partitions[[8]][[3]]
      [1] 2 2


      [[9]]$partitions[[9]]
      [[9]]$partitions[[9]][[1]]
      [1] 1

      [[9]]$partitions[[9]][[2]]
      [1] 1

      [[9]]$partitions[[9]][[3]]
      [1] 2

      [[9]]$partitions[[9]][[4]]
      [1] 2




      [[10]]
      [[10]]$multiset
      [1] 1 1 2 3

      [[10]]$partitions
      [[10]]$partitions[[1]]
      [[10]]$partitions[[1]][[1]]
      [1] 1 1 2 3


      [[10]]$partitions[[2]]
      [[10]]$partitions[[2]][[1]]
      [1] 1 1 2

      [[10]]$partitions[[2]][[2]]
      [1] 3


      [[10]]$partitions[[3]]
      [[10]]$partitions[[3]][[1]]
      [1] 1 1 3

      [[10]]$partitions[[3]][[2]]
      [1] 2


      [[10]]$partitions[[4]]
      [[10]]$partitions[[4]][[1]]
      [1] 1 1

      [[10]]$partitions[[4]][[2]]
      [1] 2 3


      [[10]]$partitions[[5]]
      [[10]]$partitions[[5]][[1]]
      [1] 1 1

      [[10]]$partitions[[5]][[2]]
      [1] 2

      [[10]]$partitions[[5]][[3]]
      [1] 3


      [[10]]$partitions[[6]]
      [[10]]$partitions[[6]][[1]]
      [1] 1 2 3

      [[10]]$partitions[[6]][[2]]
      [1] 1


      [[10]]$partitions[[7]]
      [[10]]$partitions[[7]][[1]]
      [1] 1 2

      [[10]]$partitions[[7]][[2]]
      [1] 1 3


      [[10]]$partitions[[8]]
      [[10]]$partitions[[8]][[1]]
      [1] 1 2

      [[10]]$partitions[[8]][[2]]
      [1] 1

      [[10]]$partitions[[8]][[3]]
      [1] 3


      [[10]]$partitions[[9]]
      [[10]]$partitions[[9]][[1]]
      [1] 1 3

      [[10]]$partitions[[9]][[2]]
      [1] 1

      [[10]]$partitions[[9]][[3]]
      [1] 2


      [[10]]$partitions[[10]]
      [[10]]$partitions[[10]][[1]]
      [1] 1

      [[10]]$partitions[[10]][[2]]
      [1] 1

      [[10]]$partitions[[10]][[3]]
      [1] 2 3


      [[10]]$partitions[[11]]
      [[10]]$partitions[[11]][[1]]
      [1] 1

      [[10]]$partitions[[11]][[2]]
      [1] 1

      [[10]]$partitions[[11]][[3]]
      [1] 2

      [[10]]$partitions[[11]][[4]]
      [1] 3




      [[11]]
      [[11]]$multiset
      [1] 1 1 1 1 1

      [[11]]$partitions
      [[11]]$partitions[[1]]
      [[11]]$partitions[[1]][[1]]
      [1] 1 1 1 1 1


      [[11]]$partitions[[2]]
      [[11]]$partitions[[2]][[1]]
      [1] 1 1 1 1

      [[11]]$partitions[[2]][[2]]
      [1] 1


      [[11]]$partitions[[3]]
      [[11]]$partitions[[3]][[1]]
      [1] 1 1 1

      [[11]]$partitions[[3]][[2]]
      [1] 1 1


      [[11]]$partitions[[4]]
      [[11]]$partitions[[4]][[1]]
      [1] 1 1 1

      [[11]]$partitions[[4]][[2]]
      [1] 1

      [[11]]$partitions[[4]][[3]]
      [1] 1


      [[11]]$partitions[[5]]
      [[11]]$partitions[[5]][[1]]
      [1] 1 1

      [[11]]$partitions[[5]][[2]]
      [1] 1 1

      [[11]]$partitions[[5]][[3]]
      [1] 1


      [[11]]$partitions[[6]]
      [[11]]$partitions[[6]][[1]]
      [1] 1 1

      [[11]]$partitions[[6]][[2]]
      [1] 1

      [[11]]$partitions[[6]][[3]]
      [1] 1

      [[11]]$partitions[[6]][[4]]
      [1] 1


      [[11]]$partitions[[7]]
      [[11]]$partitions[[7]][[1]]
      [1] 1

      [[11]]$partitions[[7]][[2]]
      [1] 1

      [[11]]$partitions[[7]][[3]]
      [1] 1

      [[11]]$partitions[[7]][[4]]
      [1] 1

      [[11]]$partitions[[7]][[5]]
      [1] 1




      [[12]]
      [[12]]$multiset
      [1] 1 1 1 1 2

      [[12]]$partitions
      [[12]]$partitions[[1]]
      [[12]]$partitions[[1]][[1]]
      [1] 1 1 1 1 2


      [[12]]$partitions[[2]]
      [[12]]$partitions[[2]][[1]]
      [1] 1 1 1 1

      [[12]]$partitions[[2]][[2]]
      [1] 2


      [[12]]$partitions[[3]]
      [[12]]$partitions[[3]][[1]]
      [1] 1 1 1 2

      [[12]]$partitions[[3]][[2]]
      [1] 1


      [[12]]$partitions[[4]]
      [[12]]$partitions[[4]][[1]]
      [1] 1 1 1

      [[12]]$partitions[[4]][[2]]
      [1] 1 2


      [[12]]$partitions[[5]]
      [[12]]$partitions[[5]][[1]]
      [1] 1 1 1

      [[12]]$partitions[[5]][[2]]
      [1] 1

      [[12]]$partitions[[5]][[3]]
      [1] 2


      [[12]]$partitions[[6]]
      [[12]]$partitions[[6]][[1]]
      [1] 1 1 2

      [[12]]$partitions[[6]][[2]]
      [1] 1 1


      [[12]]$partitions[[7]]
      [[12]]$partitions[[7]][[1]]
      [1] 1 1 2

      [[12]]$partitions[[7]][[2]]
      [1] 1

      [[12]]$partitions[[7]][[3]]
      [1] 1


      [[12]]$partitions[[8]]
      [[12]]$partitions[[8]][[1]]
      [1] 1 1

      [[12]]$partitions[[8]][[2]]
      [1] 1 1

      [[12]]$partitions[[8]][[3]]
      [1] 2


      [[12]]$partitions[[9]]
      [[12]]$partitions[[9]][[1]]
      [1] 1 1

      [[12]]$partitions[[9]][[2]]
      [1] 1 2

      [[12]]$partitions[[9]][[3]]
      [1] 1


      [[12]]$partitions[[10]]
      [[12]]$partitions[[10]][[1]]
      [1] 1 1

      [[12]]$partitions[[10]][[2]]
      [1] 1

      [[12]]$partitions[[10]][[3]]
      [1] 1

      [[12]]$partitions[[10]][[4]]
      [1] 2


      [[12]]$partitions[[11]]
      [[12]]$partitions[[11]][[1]]
      [1] 1 2

      [[12]]$partitions[[11]][[2]]
      [1] 1

      [[12]]$partitions[[11]][[3]]
      [1] 1

      [[12]]$partitions[[11]][[4]]
      [1] 1


      [[12]]$partitions[[12]]
      [[12]]$partitions[[12]][[1]]
      [1] 1

      [[12]]$partitions[[12]][[2]]
      [1] 1

      [[12]]$partitions[[12]][[3]]
      [1] 1

      [[12]]$partitions[[12]][[4]]
      [1] 1

      [[12]]$partitions[[12]][[5]]
      [1] 2




      [[13]]
      [[13]]$multiset
      [1] 1 1 1 2 2

      [[13]]$partitions
      [[13]]$partitions[[1]]
      [[13]]$partitions[[1]][[1]]
      [1] 1 1 1 2 2


      [[13]]$partitions[[2]]
      [[13]]$partitions[[2]][[1]]
      [1] 1 1 1 2

      [[13]]$partitions[[2]][[2]]
      [1] 2


      [[13]]$partitions[[3]]
      [[13]]$partitions[[3]][[1]]
      [1] 1 1 1

      [[13]]$partitions[[3]][[2]]
      [1] 2 2


      [[13]]$partitions[[4]]
      [[13]]$partitions[[4]][[1]]
      [1] 1 1 1

      [[13]]$partitions[[4]][[2]]
      [1] 2

      [[13]]$partitions[[4]][[3]]
      [1] 2


      [[13]]$partitions[[5]]
      [[13]]$partitions[[5]][[1]]
      [1] 1 1 2 2

      [[13]]$partitions[[5]][[2]]
      [1] 1


      [[13]]$partitions[[6]]
      [[13]]$partitions[[6]][[1]]
      [1] 1 1 2

      [[13]]$partitions[[6]][[2]]
      [1] 1 2


      [[13]]$partitions[[7]]
      [[13]]$partitions[[7]][[1]]
      [1] 1 1 2

      [[13]]$partitions[[7]][[2]]
      [1] 1

      [[13]]$partitions[[7]][[3]]
      [1] 2


      [[13]]$partitions[[8]]
      [[13]]$partitions[[8]][[1]]
      [1] 1 1

      [[13]]$partitions[[8]][[2]]
      [1] 1 2 2


      [[13]]$partitions[[9]]
      [[13]]$partitions[[9]][[1]]
      [1] 1 1

      [[13]]$partitions[[9]][[2]]
      [1] 1 2

      [[13]]$partitions[[9]][[3]]
      [1] 2


      [[13]]$partitions[[10]]
      [[13]]$partitions[[10]][[1]]
      [1] 1 1

      [[13]]$partitions[[10]][[2]]
      [1] 1

      [[13]]$partitions[[10]][[3]]
      [1] 2 2


      [[13]]$partitions[[11]]
      [[13]]$partitions[[11]][[1]]
      [1] 1 1

      [[13]]$partitions[[11]][[2]]
      [1] 1

      [[13]]$partitions[[11]][[3]]
      [1] 2

      [[13]]$partitions[[11]][[4]]
      [1] 2


      [[13]]$partitions[[12]]
      [[13]]$partitions[[12]][[1]]
      [1] 1 2 2

      [[13]]$partitions[[12]][[2]]
      [1] 1

      [[13]]$partitions[[12]][[3]]
      [1] 1


      [[13]]$partitions[[13]]
      [[13]]$partitions[[13]][[1]]
      [1] 1 2

      [[13]]$partitions[[13]][[2]]
      [1] 1 2

      [[13]]$partitions[[13]][[3]]
      [1] 1


      [[13]]$partitions[[14]]
      [[13]]$partitions[[14]][[1]]
      [1] 1 2

      [[13]]$partitions[[14]][[2]]
      [1] 1

      [[13]]$partitions[[14]][[3]]
      [1] 1

      [[13]]$partitions[[14]][[4]]
      [1] 2


      [[13]]$partitions[[15]]
      [[13]]$partitions[[15]][[1]]
      [1] 1

      [[13]]$partitions[[15]][[2]]
      [1] 1

      [[13]]$partitions[[15]][[3]]
      [1] 1

      [[13]]$partitions[[15]][[4]]
      [1] 2 2


      [[13]]$partitions[[16]]
      [[13]]$partitions[[16]][[1]]
      [1] 1

      [[13]]$partitions[[16]][[2]]
      [1] 1

      [[13]]$partitions[[16]][[3]]
      [1] 1

      [[13]]$partitions[[16]][[4]]
      [1] 2

      [[13]]$partitions[[16]][[5]]
      [1] 2




      [[14]]
      [[14]]$multiset
      [1] 1 1 1 2 3

      [[14]]$partitions
      [[14]]$partitions[[1]]
      [[14]]$partitions[[1]][[1]]
      [1] 1 1 1 2 3


      [[14]]$partitions[[2]]
      [[14]]$partitions[[2]][[1]]
      [1] 1 1 1 2

      [[14]]$partitions[[2]][[2]]
      [1] 3


      [[14]]$partitions[[3]]
      [[14]]$partitions[[3]][[1]]
      [1] 1 1 1 3

      [[14]]$partitions[[3]][[2]]
      [1] 2


      [[14]]$partitions[[4]]
      [[14]]$partitions[[4]][[1]]
      [1] 1 1 1

      [[14]]$partitions[[4]][[2]]
      [1] 2 3


      [[14]]$partitions[[5]]
      [[14]]$partitions[[5]][[1]]
      [1] 1 1 1

      [[14]]$partitions[[5]][[2]]
      [1] 2

      [[14]]$partitions[[5]][[3]]
      [1] 3


      [[14]]$partitions[[6]]
      [[14]]$partitions[[6]][[1]]
      [1] 1 1 2 3

      [[14]]$partitions[[6]][[2]]
      [1] 1


      [[14]]$partitions[[7]]
      [[14]]$partitions[[7]][[1]]
      [1] 1 1 2

      [[14]]$partitions[[7]][[2]]
      [1] 1 3


      [[14]]$partitions[[8]]
      [[14]]$partitions[[8]][[1]]
      [1] 1 1 2

      [[14]]$partitions[[8]][[2]]
      [1] 1

      [[14]]$partitions[[8]][[3]]
      [1] 3


      [[14]]$partitions[[9]]
      [[14]]$partitions[[9]][[1]]
      [1] 1 1 3

      [[14]]$partitions[[9]][[2]]
      [1] 1 2


      [[14]]$partitions[[10]]
      [[14]]$partitions[[10]][[1]]
      [1] 1 1 3

      [[14]]$partitions[[10]][[2]]
      [1] 1

      [[14]]$partitions[[10]][[3]]
      [1] 2


      [[14]]$partitions[[11]]
      [[14]]$partitions[[11]][[1]]
      [1] 1 1

      [[14]]$partitions[[11]][[2]]
      [1] 1 2 3


      [[14]]$partitions[[12]]
      [[14]]$partitions[[12]][[1]]
      [1] 1 1

      [[14]]$partitions[[12]][[2]]
      [1] 1 2

      [[14]]$partitions[[12]][[3]]
      [1] 3


      [[14]]$partitions[[13]]
      [[14]]$partitions[[13]][[1]]
      [1] 1 1

      [[14]]$partitions[[13]][[2]]
      [1] 1 3

      [[14]]$partitions[[13]][[3]]
      [1] 2


      [[14]]$partitions[[14]]
      [[14]]$partitions[[14]][[1]]
      [1] 1 1

      [[14]]$partitions[[14]][[2]]
      [1] 1

      [[14]]$partitions[[14]][[3]]
      [1] 2 3


      [[14]]$partitions[[15]]
      [[14]]$partitions[[15]][[1]]
      [1] 1 1

      [[14]]$partitions[[15]][[2]]
      [1] 1

      [[14]]$partitions[[15]][[3]]
      [1] 2

      [[14]]$partitions[[15]][[4]]
      [1] 3


      [[14]]$partitions[[16]]
      [[14]]$partitions[[16]][[1]]
      [1] 1 2 3

      [[14]]$partitions[[16]][[2]]
      [1] 1

      [[14]]$partitions[[16]][[3]]
      [1] 1


      [[14]]$partitions[[17]]
      [[14]]$partitions[[17]][[1]]
      [1] 1 2

      [[14]]$partitions[[17]][[2]]
      [1] 1 3

      [[14]]$partitions[[17]][[3]]
      [1] 1


      [[14]]$partitions[[18]]
      [[14]]$partitions[[18]][[1]]
      [1] 1 2

      [[14]]$partitions[[18]][[2]]
      [1] 1

      [[14]]$partitions[[18]][[3]]
      [1] 1

      [[14]]$partitions[[18]][[4]]
      [1] 3


      [[14]]$partitions[[19]]
      [[14]]$partitions[[19]][[1]]
      [1] 1 3

      [[14]]$partitions[[19]][[2]]
      [1] 1

      [[14]]$partitions[[19]][[3]]
      [1] 1

      [[14]]$partitions[[19]][[4]]
      [1] 2


      [[14]]$partitions[[20]]
      [[14]]$partitions[[20]][[1]]
      [1] 1

      [[14]]$partitions[[20]][[2]]
      [1] 1

      [[14]]$partitions[[20]][[3]]
      [1] 1

      [[14]]$partitions[[20]][[4]]
      [1] 2 3


      [[14]]$partitions[[21]]
      [[14]]$partitions[[21]][[1]]
      [1] 1

      [[14]]$partitions[[21]][[2]]
      [1] 1

      [[14]]$partitions[[21]][[3]]
      [1] 1

      [[14]]$partitions[[21]][[4]]
      [1] 2

      [[14]]$partitions[[21]][[5]]
      [1] 3




      [[15]]
      [[15]]$multiset
      [1] 1 1 2 2 3

      [[15]]$partitions
      [[15]]$partitions[[1]]
      [[15]]$partitions[[1]][[1]]
      [1] 1 1 2 2 3


      [[15]]$partitions[[2]]
      [[15]]$partitions[[2]][[1]]
      [1] 1 1 2 2

      [[15]]$partitions[[2]][[2]]
      [1] 3


      [[15]]$partitions[[3]]
      [[15]]$partitions[[3]][[1]]
      [1] 1 1 2 3

      [[15]]$partitions[[3]][[2]]
      [1] 2


      [[15]]$partitions[[4]]
      [[15]]$partitions[[4]][[1]]
      [1] 1 1 2

      [[15]]$partitions[[4]][[2]]
      [1] 2 3


      [[15]]$partitions[[5]]
      [[15]]$partitions[[5]][[1]]
      [1] 1 1 2

      [[15]]$partitions[[5]][[2]]
      [1] 2

      [[15]]$partitions[[5]][[3]]
      [1] 3


      [[15]]$partitions[[6]]
      [[15]]$partitions[[6]][[1]]
      [1] 1 1 3

      [[15]]$partitions[[6]][[2]]
      [1] 2 2


      [[15]]$partitions[[7]]
      [[15]]$partitions[[7]][[1]]
      [1] 1 1 3

      [[15]]$partitions[[7]][[2]]
      [1] 2

      [[15]]$partitions[[7]][[3]]
      [1] 2


      [[15]]$partitions[[8]]
      [[15]]$partitions[[8]][[1]]
      [1] 1 1

      [[15]]$partitions[[8]][[2]]
      [1] 2 2 3


      [[15]]$partitions[[9]]
      [[15]]$partitions[[9]][[1]]
      [1] 1 1

      [[15]]$partitions[[9]][[2]]
      [1] 2 2

      [[15]]$partitions[[9]][[3]]
      [1] 3


      [[15]]$partitions[[10]]
      [[15]]$partitions[[10]][[1]]
      [1] 1 1

      [[15]]$partitions[[10]][[2]]
      [1] 2 3

      [[15]]$partitions[[10]][[3]]
      [1] 2


      [[15]]$partitions[[11]]
      [[15]]$partitions[[11]][[1]]
      [1] 1 1

      [[15]]$partitions[[11]][[2]]
      [1] 2

      [[15]]$partitions[[11]][[3]]
      [1] 2

      [[15]]$partitions[[11]][[4]]
      [1] 3


      [[15]]$partitions[[12]]
      [[15]]$partitions[[12]][[1]]
      [1] 1 2 2 3

      [[15]]$partitions[[12]][[2]]
      [1] 1


      [[15]]$partitions[[13]]
      [[15]]$partitions[[13]][[1]]
      [1] 1 2 2

      [[15]]$partitions[[13]][[2]]
      [1] 1 3


      [[15]]$partitions[[14]]
      [[15]]$partitions[[14]][[1]]
      [1] 1 2 2

      [[15]]$partitions[[14]][[2]]
      [1] 1

      [[15]]$partitions[[14]][[3]]
      [1] 3


      [[15]]$partitions[[15]]
      [[15]]$partitions[[15]][[1]]
      [1] 1 2 3

      [[15]]$partitions[[15]][[2]]
      [1] 1 2


      [[15]]$partitions[[16]]
      [[15]]$partitions[[16]][[1]]
      [1] 1 2 3

      [[15]]$partitions[[16]][[2]]
      [1] 1

      [[15]]$partitions[[16]][[3]]
      [1] 2


      [[15]]$partitions[[17]]
      [[15]]$partitions[[17]][[1]]
      [1] 1 2

      [[15]]$partitions[[17]][[2]]
      [1] 1 2

      [[15]]$partitions[[17]][[3]]
      [1] 3


      [[15]]$partitions[[18]]
      [[15]]$partitions[[18]][[1]]
      [1] 1 2

      [[15]]$partitions[[18]][[2]]
      [1] 1 3

      [[15]]$partitions[[18]][[3]]
      [1] 2


      [[15]]$partitions[[19]]
      [[15]]$partitions[[19]][[1]]
      [1] 1 2

      [[15]]$partitions[[19]][[2]]
      [1] 1

      [[15]]$partitions[[19]][[3]]
      [1] 2 3


      [[15]]$partitions[[20]]
      [[15]]$partitions[[20]][[1]]
      [1] 1 2

      [[15]]$partitions[[20]][[2]]
      [1] 1

      [[15]]$partitions[[20]][[3]]
      [1] 2

      [[15]]$partitions[[20]][[4]]
      [1] 3


      [[15]]$partitions[[21]]
      [[15]]$partitions[[21]][[1]]
      [1] 1 3

      [[15]]$partitions[[21]][[2]]
      [1] 1

      [[15]]$partitions[[21]][[3]]
      [1] 2 2


      [[15]]$partitions[[22]]
      [[15]]$partitions[[22]][[1]]
      [1] 1 3

      [[15]]$partitions[[22]][[2]]
      [1] 1

      [[15]]$partitions[[22]][[3]]
      [1] 2

      [[15]]$partitions[[22]][[4]]
      [1] 2


      [[15]]$partitions[[23]]
      [[15]]$partitions[[23]][[1]]
      [1] 1

      [[15]]$partitions[[23]][[2]]
      [1] 1

      [[15]]$partitions[[23]][[3]]
      [1] 2 2 3


      [[15]]$partitions[[24]]
      [[15]]$partitions[[24]][[1]]
      [1] 1

      [[15]]$partitions[[24]][[2]]
      [1] 1

      [[15]]$partitions[[24]][[3]]
      [1] 2 2

      [[15]]$partitions[[24]][[4]]
      [1] 3


      [[15]]$partitions[[25]]
      [[15]]$partitions[[25]][[1]]
      [1] 1

      [[15]]$partitions[[25]][[2]]
      [1] 1

      [[15]]$partitions[[25]][[3]]
      [1] 2 3

      [[15]]$partitions[[25]][[4]]
      [1] 2


      [[15]]$partitions[[26]]
      [[15]]$partitions[[26]][[1]]
      [1] 1

      [[15]]$partitions[[26]][[2]]
      [1] 1

      [[15]]$partitions[[26]][[3]]
      [1] 2

      [[15]]$partitions[[26]][[4]]
      [1] 2

      [[15]]$partitions[[26]][[5]]
      [1] 3
