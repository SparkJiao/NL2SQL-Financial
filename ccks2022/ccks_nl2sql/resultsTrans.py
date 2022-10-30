# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sqliteStructureTrans import gen_sqlite_struct_file


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_results_file = "results.xlsx"
    gen_sqlite_struct_file(test_results_file)

