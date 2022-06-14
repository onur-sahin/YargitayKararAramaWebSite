

from flask import Flask, flash, redirect, render_template, request, url_for
from createMarkedText import get_marked_text

from Search import Search
from createMarkedText import get_marked_text

search_drive = Search()

app = Flask(__name__, template_folder="./templates")
app.secret_key = "abc" 
# app.config['SECRET_KEY']

# @app.after_request
# def add_header(r):
#     r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
#     r.headers["Pragma"] = "no-cache"
#     r.headers["Expires"] = "0"
#     r.headers['Cache-Control'] = 'public, max-age=0'
#     return r

@app.route('/', methods = ['POST', 'GET'])

def MainPage():
    
    return render_template('main.html')

@app.route('/results', methods = ['POST', 'GET'])

def ResultPage():


    if( request.method == 'POST'):
        
        query_text = request.form["query_str"]

    result = search_drive.search_in_Search(query_text)

    head =  """
            <table class="table table-hover">
                    <thead>
                    <tr>
                        <th scope="col">#</th>
                        <th scope="col">ID</th>
                        <th scope="col">Daire</th>
                        <th scope="col">Esas</th>
                        <th scope="col">Karar</th>
                        <th scope="col">Tarih</th>
                    </tr>
                    </thead>
                    <tbody>
            """
    body = ""

    order = 0
    
    for id in result:

        row = result[id]

        markedText = get_marked_text(row["metin"])

        order += 1

        body += """
                    <tr>
                        <th scope="row">{}</th>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{}</td>
                        <td><button onclick="openDocument ({});">Aç</button></td>
                    </tr>
                    <tr id="tr-{}" style="Display:none;"> <td colspan="7">{}</td></tr>
                """.format( order,
                            row["id"],
                            row["daire_ismi"],
                            row["esas_no"],
                            row["karar_no"],
                            row["tarih"],
                            order,
                            order,
                            markedText)

    


    foot =  """
                    </tbody>
                </table>    
            """
    results = head + body + foot


    


    return render_template('results.html', results = results)

# @app.route("/show",  methods = ['POST', 'GET'])

# def show():

#     if( request.method == 'POST'):
    
#     id = request.form["query_str"]

#     document = "keu ylakeyualkeuyml"
    

#     return render_template("document.html", document=document)


if __name__ == '__main__':
    app.run()