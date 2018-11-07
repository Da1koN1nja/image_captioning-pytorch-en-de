"""
Functions and classes for aligning two lists using dynamic programming.

The algorithm is based on on a slight variation of the method given at:
http://www.avatar.se/molbioinfo2001/dynprog/adv_dynamic.html. By default NIST
insertion, deletion and substitution penalties are used.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2011, 2014, 2015
"""

import numpy as np
import nltk

#-----------------------------------------------------------------------------#
#                         DYNAMIC PROGRAMMING CLASSES                         #
#-----------------------------------------------------------------------------#

class DPEntry:
    """Alignment type ("d", "i", "s", or "m") and an integer score."""
    def __init__(self, align="m", score=0):
        self.align = align
        self.score = score


class DPError(object):
    """
    Attributes
    ----------
    n_del : int
    n_ins : int
    n_sub : int
    n_match : int
    n_total : int
    """

    def __init__(self, n_del=0, n_ins=0, n_sub=0, n_match=0, n_total=0):
        self.n_del = n_del
        self.n_ins = n_ins
        self.n_sub = n_sub
        self.n_match = n_match
        self.n_total = n_total

    def __add__(self, other):
        """Add this DPError to another."""
        if type(other) == DPError:
            self.n_del += other.n_del
            self.n_ins += other.n_ins
            self.n_sub += other.n_sub
            self.n_match += other.n_match
            self.n_total += other.n_total
        return self

    __radd__ = __add__
    __iadd__ = __add__

    def __str__(self):
        """Returns a string representation of the alignment error."""
        return (
            "H = " + str(self.n_match) + ", D = " + str(self.n_del) + ", S = "
            + str(self.n_sub) + ", I = " + str(self.n_ins)+ ", N = " +
            str(self.n_total)
            )

    def get_levenshtein(self):
        """Returns the Levenshtein distance of the alignment."""
        return self.n_del + self.n_sub + self.n_ins

    def get_accuracy(self):
        """
        Calculates the accuracy given the stored errors using the formula:
        Accuracy = (Matches - Insertions) / Total
        """
        return float(self.n_match - self.n_ins) / self.n_total

    def get_wer(self):
        """
        Calculates the word error rate (WER) using:
        WER = (Substitutions + Deletions + Insertions) / Total
        """
        return float(self.n_sub + self.n_del + self.n_ins) / self.n_total


#-----------------------------------------------------------------------------#
#                    DYNAMIC PROGRAMMING ALIGNMENT FUNCTION                   #
#-----------------------------------------------------------------------------#

def dp_align(ref_list, test_list, ins_penalty=3, del_penalty=3, sub_penalty=4):
    """
    Performs dynamic programming alignment of `ref_list` to `test_list`.

    Parameters
    ----------
    ref_list : list
    test_list : list
    """

    # Initialise the alignment matrix
    dp_matrix = np.empty([len(test_list) + 1, len(ref_list) + 1], dtype = object)
    for i in range(len(test_list) + 1):
        for j in range(len(ref_list) + 1):
            dp_matrix[i][j] = DPEntry()

    # Initialise the originf
    dp_matrix[0][0].score = 0
    dp_matrix[0][0].align = "m"

    # The first row is all delections:
    for j in range(1, len(ref_list) + 1):
        dp_matrix[0][j].score = j*del_penalty
        dp_matrix[0][j].align = "d"

    # Fill dp_matrix
    for i in range(1, len(test_list) + 1):

        # First column is all insertions
        dp_matrix[i][0].score = i*ins_penalty
        dp_matrix[i][0].align = "i"

        for j in range(1, len(ref_list) + 1):
            del_score = dp_matrix[i, j - 1].score + del_penalty
            ins_score = dp_matrix[i - 1, j].score + ins_penalty

            if test_list[i - 1] == ref_list[j - 1]:

                # Considering a match
                match_score = dp_matrix[i - 1, j - 1].score

                # Test for a match
                if match_score <= del_score and match_score <= ins_score:
                    dp_matrix[i, j].score = match_score
                    dp_matrix[i, j].align = "m"
                # Test for a deletion
                elif del_score <= ins_score:
                    dp_matrix[i, j].score = del_score
                    dp_matrix[i, j].align = "d"
                # Test for an insertion (only option left)
                else:
                    dp_matrix[i, j].score = ins_score
                    dp_matrix[i, j].align = "i"

            else:

                # Considering a substitution
                sub_score = dp_matrix[i - 1, j - 1].score + sub_penalty

                # Test for a substitution
                if sub_score < del_score and sub_score <= ins_score:
                    dp_matrix[i, j].score = sub_score
                    dp_matrix[i, j].align = "s"
                # Test for a deletion
                elif del_score <= ins_score:
                    dp_matrix[i, j].score = del_score
                    dp_matrix[i, j].align = "d"
                # Test for an insertion (only option left)
                else:
                    dp_matrix[i, j].score = ins_score
                    dp_matrix[i, j].align = "i"

    # Perform alignment by tracking through the dp_matrix
    dp_errors = DPError()
    dp_errors.n_total = len(ref_list)
    i = len(test_list)
    j = len(ref_list)
    while i > 0 or j > 0:
        if dp_matrix[i, j].align == "m":
            #print test_list[i - 1], ref_list[j - 1]
            i -= 1
            j -= 1
            dp_errors.n_match += 1
        elif dp_matrix[i, j].align == "s":
            #print test_list[i - 1], ref_list[j - 1]
            i -= 1
            j -= 1
            dp_errors.n_sub += 1
        elif dp_matrix[i, j].align == "d":
            #print "-", ref_list[j - 1]
            j -= 1
            dp_errors.n_del += 1
        elif dp_matrix[i, j].align == "i":
            #print test_list[i - 1], "-"
            i -= 1
            dp_errors.n_ins += 1

    # Return the alignment results
    
    return dp_errors


def loop_values(ref_list, test):
    WER_score = {}
    lev_score = {}
    for i in range(len(ref_list)):
        current_object = dp_align(ref_list[i], test, ins_penalty=1, del_penalty=1, sub_penalty=1)
        #print current_object
        locale = "WER of reference " + str(i)                   
        WER_score[locale] = str(current_object.get_wer())
        locale = "LEV of reference " + str(i)                   
        lev_score[locale] = str(current_object.get_levenshtein())
        #print current_object
    return WER_score, lev_score

def system_wer_score(ref, test):
    system_lev = 0.0
    system_total = 0.0
    system_wer = 0.0
    
    #print(len(test))
    #print(test)
    #print(len(ref))
    for i in range(len(test)):       
        current_list = ref[i]
        #print(current_list)
        min_wer = 100
        
        for j in range(len(current_list)):            
            #print(test[i])
            current_matrix = dp_align(current_list[j], test[i], ins_penalty=1, del_penalty=1, sub_penalty=1)            
            #current_lev = current_matrix.get_levenshtein()
            #current_total = current_matrix.n_total
            current_wer = current_matrix.get_wer()
            if current_wer < min_wer:
                min_wer = current_wer
            
            #system_lev += current_matrix.get_levenshtein()
            #system_total += current_matrix.n_total
            
        system_wer += min_wer
    
    #print(system_wer)
    #print(len(ref))
    return system_wer/len(ref)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    a = dp_align("recycling", "recycle", ins_penalty=1, del_penalty=1, sub_penalty=1)
    print ("Levenshtein distance between recycling and recycle: " + str(a.get_levenshtein()))
    print ("WER  of two words: " + str(a.get_wer()))

    ref_list_1 = [["The", "cat", "is", "on", "the", "mat"], ["The", "cat", "sits", "on", "the", "mat"], ["The", "mat", "is", "under", "the", "cat"]]
    #ref_list_2 = [["The" "boy" "kicks" "the", "ball"]
    
    test_1 = ["The", "cat", "sits", "in", "the", "mat"]
    #test_2
    print (len(ref_list_1))
    #WER_score = {}
    #lev_score = {}
    #for i in range(len(ref_list_1)):
    #               current_object = dp_align(ref_list_1[i], test_1, ins_penalty=1, del_penalty=1, sub_penalty=1)
    #               locale = "WER of reference " + str(i)                   
    #               WER_score[locale] = str(current_object.get_wer())
    #               locale = "LEV of reference " + str(i)                   
    #               lev_score[locale] = str(current_object.get_levenshtein())
    #               print current_object

    #print WER_score
    #print lev_score

    train1 = "The boy kicks the ball"
    train2 = "The boy kicks the ball across the field"
    train3 = "The boy kicks the soccer ball"
    
    test2 = "The boy hits the ball into the air"

    ref_list_2 = [nltk.word_tokenize(train1), nltk.word_tokenize(train2), nltk.word_tokenize(train3)]
    test2_split = nltk.word_tokenize(test2)
    
    #print WER_score
    #print lev_score

    ref_tot_list = [ref_list_1, ref_list_2]
    test_list = [test_1, test2_split]
    WER_score_list = {}
    lev_score_list = {}
    for i in range(len(ref_tot_list)):
        #print "Reference \{}: " + str(i)
        WER_score, lev_score = loop_values(ref_tot_list[i], test_list[i])
        
       
        WER_score_list["Image " + str(i)] = WER_score
        lev_score_list["Image " + str(i)] = lev_score
    #print WER_score_list
    #print "testing"
    #print type(WER_score_list["Image "+str(0)]["WER of reference " + str(0)])
    #print lev_score_list
    accuracy = 0
    iterations = 0
    for i in range(len(WER_score_list)):
        for j in range(len(WER_score_list["Image "+str(i)])):
            accuracy += (1 - float(WER_score_list["Image "+str(i)]["WER of reference " + str(j)]))
            iterations += 1
    print (str(accuracy/iterations))
    print(a.n_total)
if __name__ == "__main__":
    main()
