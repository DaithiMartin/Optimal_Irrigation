## Virtual Environment

---
These nifty little things keep your system environment separate from you working environment. 
This also allows you to port your project and its dependencies easily between machines. Beautiful! 
Use a virtual environment unless you have a strong compelling reason not to.

Pycharm will build one automatically when starting a new project but if you need to build one manually:

~/ python3 -m venv name_of_venv_directory

Commonly:

~/ python3 -m venv venv

Activating your virtual environment:

~/ source venv/bin/activate

To deactivate:

~/ deactivate

For reasons that will become clear shortly, we generally want to add this venv directory to our .gitignore file. 
More on git after we cover dependencies.

## Dependencies

---
For deployment of your project across multiple enviroments, virtural or native, it is very helpful to have a dependency list.
Creating this manually is tedious at best but thankfully pip has a beautiful tool to caputre youre current dependencies,
and pipe them into a requirements.txt file.


(venv) ~/ pip freeze > requirements.txt

Absolutely amazing! Now when you want pip to install these.

(venv) ~/ pip install -r requirements.txt

## git

---

Git repositories are directed acyclic graphs used to store the state of you project at each node. Simply put it is a 
version control system that allows you to back up to previous states. 

To initialize in current working directory:

(venv) ~/ git init -b main

Adding files to .gitignore so that they are not tracked:

(venv) ~/ echo "file_or_directory" >> .gitignore

Here we use the ">>" as opposed to the ">" in the pip freeze command. One pipe ">" overwrites the entire file and a 
double pipe ">>" will add to file. 

It is generally good to add the "venv/" directory to the .gitignore and just handle new build by initializing a new 
virtual environment and using pip to install the dependencies from requirements.txt

Add any remaining files to tracked files

(venv) ~/ git add .

If you happen to add a file to tracked files and want to remove it:

(venv) ~/ git rm -r --cached

** this does not appear to be working exactly as intended. When tried on venv directory it removed pip functionality as well.
this forced me to delete the venv and initialize a new one **
