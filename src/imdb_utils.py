from imdb import IMDb
import re
import multiprocessing as mp
import string

ia = IMDb()

# returns the IMDb movie ID
def get_movie_ID(movie_name):
    movie = ia.search_movie(movie_name)[0]
    return movie.movieID

# returns a list of movie IDs where all of the actors in 'actors_names' participated
def get_movies_with_actors(actors_names, verbose=False):
    if verbose:
        print("[BUS-PEL] Buscando películas donde actuen: ", end='')
        print(*actors_names, sep=', ', end='')
        print("...")

    res = []
    
    actors = []
    for actor_name in actors_names:
        actor_id = ia.search_person(actor_name)[0].getID()
        actors.append(ia.get_person(actor_id))
        
    movies= []
    for actor in actors:
        if 'actor' in actor['filmography'].keys():
            key = 'actor'
        else:
            key = 'actress'

        movies.append(actor['filmography'][key])
        
    coincidences = set(movies[0]).intersection(*movies)

    for c in coincidences:
        res.append(c.getID())

    if verbose:
        print("[BUS-PEL] Películas encontradas: ", end='')
        print(*res, sep=', ')
    
    return res

def get_movie_parallel(id):
    return ia.get_movie(id)

def get_plots(movies_ids, verbose=False):
    if verbose:
        print("[PLOT] Obteniendo plots...")

    res = []
    pool = mp.Pool(movies_ids.__len__())

    movies = []
    for movie_id in movies_ids:
        movies.append(pool.apply_async(get_movie_parallel, [movie_id]))

    pool.close()
    pool.join()

    for movie in movies:
        # get plots
        plots = movie.get().get('plot')

        if plots.__len__() > 0:
            doc = ""
            for plot in plots:
                doc += preprocess_plot(plot)

            synopsis = movie.get().get('synopsis')

            if synopsis is not None:
                doc += preprocess_plot(synopsis[0])

            res.append([doc])
    
    if verbose:
        print("[PLOT] Plots obtenidos.")
    
    return res

# removes punctuation and author from plots
def preprocess_plot(plot):
    idx = plot.find('::')

    if idx != -1:
        plot = plot[0:plot.index('::')]

    plot = re.sub("\'\w+ ", " ", plot)
    plot = plot.translate(plot.maketrans('', '', string.punctuation + string.digits))
    
    return plot