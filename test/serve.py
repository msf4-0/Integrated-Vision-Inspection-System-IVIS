# %%
from io import StringIO
import json
from logging import Handler


class Object:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__
# print json.dumps(some_big_object, default=dumper, indent=2)


me = Object()
me.name = "Onur"
me.age = 35
me.dog = Object()
me.dog.name = "Apollo"

print(me.toJSON())

data = {
    "president": {
        "name": "Zaphod Beeblebrox",
        "species": "Betelgeusian"
    }
}
print(json.dumps(data, indent=4, default=dumper))
# %%

json_string = """
{
    "researcher": {
        "name": "Ford Prefect",
        "species": "Betelgeusian",
        "relatives": [
            {
                "name": "Zaphod Beeblebrox",
                "species": "Betelgeusian"
            }
        ]
    }
}
"""
data = json.loads(json_string)
print(data)

# %%
import pandas as pd
import json

df = pd.DataFrame(
    [["a", "b"], ["c", "d"]],
    index=["row 1", "row 2"],
    columns=["col 1", "col 2"],
)
print(df)

result = df.to_json(orient="columns", indent=4)  # convert to JSON string
print(result)
parsed = json.loads(result)
print(parsed)
data2 = json.dumps(parsed, indent=4)
print(data2)

# %%
import streamlit
import io

test = io.BytesIO(b"1234")
type(test)
f = io.BytesIO(b"some initial binary data: \x00\x01")
# %%
import http.server
import socketserver
import socket
import threading
# from requests import get

# ip = get('https://api.ipify.org').text
# print('My public IP address is: {}'.format(ip))

hostname = socket.gethostname()
IP = socket.gethostbyname(hostname)


HOST, PORT = "localhost", 8000
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer((HOST, PORT), Handler) as server:
    print("Serving at PORT ", PORT)
    print("Your computer name is: ", hostname)
    print("Your computer IP Address is: ", IP)
    print("Local URL: " + f"http://localhost:{PORT}")
    server.serve_forever()
# %%
import socketserver


class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        # just send back the same data, but upper-cased
        self.request.sendall(self.data.upper())


if __name__ == "__main__":
    HOST, PORT = "localhost", 8001

    # Create the server, binding to localhost on port 9999
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()

# %%
import json

with open("/home/rchuzh/programming/image_labelling_shrdc/resources/tasks.json") as f:
    data = json.load(f)

print(data)
datas = json.dumps(data, indent=4)
print(datas)
# %%
# only runs in >=Python 3.7
from dataclasses import dataclass, field
from typing import List, TypeVar, Sequence


RANKS = '2 3 4 5 6 7 8 9 10 J Q K A'.split()
SUITS = '♣ ♢ ♡ ♠'.split()


def make_french_deck():
    return [PlayingCard(r, s) for s in SUITS for r in RANKS]


@dataclass
class PlayingCard:
    rank: str
    suit: str

    def __str__(self):
        return f'{self.suit}{self.rank}'


@dataclass
class Deck:
    cards: List[PlayingCard] = field(default_factory=make_french_deck)

    def __repr__(self):
        cards = ', '.join(f'{c!s}' for c in self.cards)
        return f'{self.__class__.__name__}({cards})'


# Deck(♣2, ♣3, ♣4, ♣5, ♣6, ♣7, ♣8, ♣9, ♣10, ♣J, ♣Q, ♣K, ♣A, ♢
# 2, ♢3, ♢4, ♢5, ♢6, ♢7, ♢8, ♢9, ♢10, ♢J, ♢Q, ♢K, ♢A,
# ♡2, ♡3, ♡4, ♡5, ♡6, ♡7, ♡8, ♡9, ♡10, ♡J, ♡Q, ♡K, ♡A, ♠2,
# ♠3, ♠4, ♠5, ♠6, ♠7, ♠8, ♠9, ♠10, ♠J, ♠Q, ♠K, ♠A)
# queen_of_hearts = PlayingCard('Q', 'Hearts')
# queen_of_hearts.rank
# queen_of_hearts == PlayingCard('Q', 'Hearts')  # check is class is correct
# type(queen_of_hearts)
# queen_of_hearts = PlayingCard('Q', 'Hearts')
# ace_of_spades = PlayingCard('A', 'Spades')
# two_cards = Deck([queen_of_hearts, ace_of_spades])
# two_cards.cards[0].rank

#------------------------------------------------------------#
# %%
T = TypeVar('T', int, float)


def vec2(x: T, y: T) -> List[T]:
    return [x, y]


def keep_positives(vector: Sequence[T]) -> List[T]:
    return[item for item in vector if item > 0]


vec2([1, 2)
keep_positives([1, 2, 3, 4])
keep_positives([-1, -2, -1, 2, 34])
# %%
import cv2
import base64
def ndarray_to_b64(ndarray):
    """
    converts a np ndarray to a b64 string readable by html-img tags
    """
    img= cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
    _, buffer= cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# %%----------------------------------------------------------------
from base64 import b64encode, decode
import io
bina= io.BytesIO(b'I want to solve import issue')
bb= bina.read()
b64code= b64encode(bb).decode('utf-8')
print(b64code)
data_url= f'data:image/jpg;base64,{b64code}'
print(f"\"{data_url}\"")
