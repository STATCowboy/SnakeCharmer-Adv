#
# Author: Jamey Johnston
# Title: Code Like a Snake Charmer: Advanced Data Modeling in Python!
# Date: 2019/11/08
# Blog: http://www.STATCowboy.com
# Twitter: @STATCowboy
# Git Repo: https://github.com/STATCowboy/SnakeCharmer-Adv
#

class Database:
    def __init__(self, instance):
        self.instance = instance

    def mssql(self, version):
        print(self.instance + " is MSSQL version " + version + " and is a Big Data Cluster!")

    def oracle(self):
        print(self.instance + " is Oracle and has crashed! Call Larry!")
 
def main():
    prod = Database("prod")
    prod.oracle()
    dev = Database("dev")
    dev.mssql("2019")

if __name__ == "__main__":
  main()