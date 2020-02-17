How to run the code:
To run the code, you only need to change the file paths.


Side note:
We attached our modified train files (replace words that appear less than three times with #UNK#) in each dataset folder.

To save time, you can just use the modified files directly instead of re-generating these files. Thus, we have
commented out the related codes in function 'def modify_train_data()'.

If you want to test the functionality, you can uncomment the codes between
print("-------- Modifying the train data (comment this part if you have modified file, for saving time.) ---------")
and
print("-------- Train data is modified. (comment this part if you have modified file, for saving time.) ---------")
