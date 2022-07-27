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
