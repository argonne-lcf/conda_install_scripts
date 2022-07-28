#!/bin/bash

# for site in $(echo /soft/datascience/conda/2022-07-19/*)
# do
#     # for file in $(find $site -type f -exec stat -c '%a' '{}' +)
#     # do
#     #     if [[ $file != *444 ]]
#     #     then
#     #         echo "$site/$file permissions are wrong"
#     #     else
#     #         echo "$site/$file permissions are correct"
#     #     fi
#     # done


#     for dir in $(find $site -type d -exec stat -c '%a' '{}' +)
#     do
#         if [[ $dir == 2500 ]]
#         then
# 	    echo $dir
#             echo "$site directory permissions are wrong"
#         else
#             echo "$site directory permissions are correct"
#         fi
#     done
#  done


site='/soft/datascience/conda/2022-07-19/'
cd "$site" && \
find . -type d ! -name .   -perm 2500 -exec chmod 2555 {} +
   # \( -type f             -perm 444 -exec $p "$site/%s file permissions are correct.\n"      {} + \) -o \
   # \( -type f           ! -perm 444 -exec $p "$site/%s file permissions are wrong.\n"        {} + \) -o \
   #\( -type d ! -name . ! -perm 555 -exec $p "$site/%s directory permissions are wrong.\n"   {} + \) -o \
     #\( -type d ! -name .   -perm 2500 -exec chmod 2555 {} + \)
# https://unix.stackexchange.com/questions/197824/what-is-the-difference-between-find-and-find-print
# KGF: enclosing the one line in \( \) somehow causes find to print out every? folder and file??
# -print is the default action if there is no "action" predicate in the expression list
# (vs. filter, condition predicates)


# https://askubuntu.com/questions/339015/what-does-mean-in-the-find-command
# \; closing delimiter to "exec cmd {} \;" runs the command once per filename
# + closing delimiter to "exec cmd {} +":
# This variant of the -exec action runs the specified command on the selected files, but the command line is built by appending each selected file name at the end; the total number of invocations of the command will be much less than the number of matched files.  The command line is built in much the same way that xargs builds its command lines.  Only one instance of `{}' is allowed within the command, and it must appear at the end, immediately before the `+'; it needs to be escaped (with a `\') o

# https://www.baeldung.com/linux/find-exec-command
