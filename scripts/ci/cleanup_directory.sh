#!/bin/bash

delete_recursive() {
    local folder="$1"

    xrdfs root://eosuser.cern.ch// ls -l "$folder" | while read line; do
        if [ "${line:0:1}" == "d" ]; then
            sub_folder=$(echo "$line" | awk '{print $7}')
            delete_recursive "$sub_folder"
            xrdfs root://eosuser.cern.ch// rmdir "$sub_folder"
        else
            file=$(echo "$line" | awk '{print $7}')
            xrdfs root://eosuser.cern.ch// rm "$file"
        fi
    done
}

all_folders=$(xrdfs root://eosuser.cern.ch// ls -l "$1" | grep '^d' | sort -k 5 -r | awk '{print $7}')
newest_folders=$(echo "$all_folders" | head -n "$2")

for folder in $all_folders; do
    if ! [[ ${newest_folders[@]} =~ ${folder} ]]; then
        delete_recursive "$folder"
        xrdfs root://eosuser.cern.ch// rmdir "$folder"
    fi
done