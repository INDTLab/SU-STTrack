import pickle

f = open('itchat.pkl','rb')
data = pickle.load(f)
print(data)
