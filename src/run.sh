#conda activate graphgan
#python main_movie.py | tee -a ../results/movie-cross.txt
#python main_movie.py | tee -a ../results/movie-cross.txt
#python main_movie.py | tee -a ../results/movie-cross.txt
#
#python main_book.py | tee -a ../results/book-cross.txt
#python main_book.py | tee -a ../results/book-cross.txt
#python main_book.py | tee -a ../results/book-cross.txt
#
#python main_music.py | tee -a ../results/music-cross.txt
#python main_music.py | tee -a ../results/music-cross.txt
#python main_music.py | tee -a ../results/music-cross.txt





#python main_movie.py --att True | tee -a ../results/movie-att.txt
#python main_movie.py --att True | tee -a ../results/movie-att.txt
#python main_movie.py --att True | tee -a ../results/movie-att.txt
#
#python main_book.py --att True | tee -a ../results/book-att.txt
#python main_book.py --att True | tee -a ../results/book-att.txt
#python main_book.py --att True | tee -a ../results/book-att.txt
#
#python main_music.py --att True | tee -a ../results/music-att.txt
#python main_music.py --att True | tee -a ../results/music-att.txt
#python main_music.py --att True | tee -a ../results/music-att.txt

python main_yelp.py --att True | tee -a ../results/yelp-att.txt
python main_yelp.py --att True | tee -a ../results/yelp-att.txt
python main_yelp.py --att True | tee -a ../results/yelp-att.txt

python main_yelp.py | tee -a ../results/yelp-cross.txt
python main_yelp.py | tee -a ../results/yelp-cross.txt
python main_yelp.py | tee -a ../results/yelp-cross.txt