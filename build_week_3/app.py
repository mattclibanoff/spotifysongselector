import pandas as pd
from flask import Flask, render_template, request
import sys
#sys.path.insert(1,"BUILD_WEEK_3/build_week_3")
from .Spotify_model import song_suggester, song, get_x, get_list
import joblib 



'build_week_3/nn' = joblib.load('model.z')
'build_week_3/enc' = joblib.load('encoder.z')
X = get_x()

def create_app():
    app = Flask(__name__)
    
    @app.route("/", methods=["GET", "POST"])
    
    def main_page():
        """
        1. Asks for your name.
        2. Greets you personally.
        3. Takes your song input.
        """
        if request.method == "GET":
            return render_template('home.html')
        if request.method == "POST":
            return render_template('greet.html',
                                   name=request.form.get("name", "you"))
            
    @app.route("/music", methods=["GET", "POST"])
    def input():
        """
        Inputs a song you like.
        Returns songs just like it!
        """
        #tried to make this "GET" method work but ran out of time
        if request.method == "GET":
            track_artist = get_list()
            return render_template('input_song.html', data=track_artist)
        if request.method == "POST":
            input = request.form.get("input_song")
            input_encoded = enc.fit_transform([input])[0]
            song_returned = song_suggester(X[input_encoded])
            return render_template('output_song.html',
                                  input_song=input, recommended_song=song_returned)
    
    return app
