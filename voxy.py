from collections import Counter
import datetime
import matplotlib.pyplot as plt
from nltk import SnowballStemmer, WordNetLemmatizer
import numpy as np
import pandas as pd
import re
import time
import urllib.request as url

import sklearn
from sklearn.decomposition import NMF
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

path1 = 'files/vox_links'
path2 = 'files/vox_pickle'
path3 = 'files/model.pkl'
path4 = 'files/alone.pkl'
path5 = 'files/together.pkl'
path6 = 'files/search_engine_tfidf_vectorizer'
path7 = 'files/search_engine_tfidf_vectorized.pkl'
# Set Global Variables
vox_links = pd.read_pickle(path1)
vox = pd.read_pickle(path2)
X_train = None
X_test = None
y_train = None
y_test = None
new_links_n = None


class LinkScraper():
    """Scrapes new links from Vox search pages"""
    def __init__(self, start_page, stop_page):
        """Start/Stop are references to Vox/news search page numbers"""
        self.start_page = start_page
        self.stop_page = stop_page

    def scrape(self, demo=True):
        """Scrape links"""
        global vox_links, new_links_n
        news_links = set(vox_links.link.values)

        # Set the loop iteration for the news pages
        news_iter = ["/" + str(_) for _ in range(self.start_page,
                                                 self.stop_page+1)]

        # Retrieve news_links from each of the news_pages.
        for i in news_iter:
            news_page = news_page = "http://www.vox.com/news" + i
            with url.urlopen(news_page) as f:
                news_read = str(f.read())
            news_pattern = r'<a href="(http://www.vox.com[\S]+2016\/[\S]+)">'
            links = set(re.findall(news_pattern, news_read))   # links found
            new = links.difference(news_links)   # new links found
            if len(new) == 0:   # if you didn't find any new links
                if demo == False:
                    print('finished at', news_page)
                break   # if you didn't find any new links on the page, break

            for link in new:   # for each new link
                news_links.add(link)   # add to our set
                vox_links = vox_links.append({'link': link}, ignore_index=True)

            time.sleep(np.random.uniform(.05, .75))   # wait before call

        new_links_n = sum(vox_links.date.isnull())
        # Set the dates for all of the links
        vox_links.date = vox_links.link.apply(self._link_date)

        if demo:
            print("While you're waiting, you might be curious to know \
                that today's lucky number is {}.".format(new_links_n))
        else:
            print(new_links_n, 'new links added')
            print(len(vox_links), 'total links')
            print(len(vox_links[vox_links.date != 'UNKNOWN']),
                  "successful date extractions")

        # Set the dates for all of the links
        vox_links.date = vox_links.link.apply(self._link_date)

        # Save the dataframe
        vox_links.to_pickle(path1)
        return

    def _link_date(self, link):
        """Obtain the article date from the link"""
        date_pattern = r"\/(201\d\/\d{1,2}\/\d{1,2})"
        link_date = re.findall(date_pattern, link)
        link_date = 'UNKNOWN' if link_date == [] else link_date[0]
        return link_date


class ArticleRetriever():
    """Download each article from Vox"""
    def __init__(self, number):
        """Set number of articles to retrieve at a time"""
        self.number = number

    def retrieve(self, demo=True):
        """Retrieve articles"""
        global vox_links
        # Rows with links that we haven't downloaded yet
        candidates = vox_links['link'][vox_links.file.isnull()]
        candidates_n = len(candidates)
        if candidates_n == 0:
            return
        elif candidates_n < self.number:
            update = candidates.values.tolist()
        else:
            update = candidates.head(self.number).values.tolist()

        # Save each story (article) as a separate file.
        for link in update:
            y = vox_links[vox_links.link == link].index.values[0]
            with url.urlopen(link) as f:   # Open link
                story_read = str(f.read())

            with open('files/'+str(y)+'.txt', 'w') as f:   # Write story
                f.write(story_read)

            vox_links.loc[y, 'file'] = y   # Set cell with the file name.

            time.sleep(np.random.uniform(.05, .75))   # wait before call

        # Save the dataframe
        if demo:
            pass
        else:
            print(sum(vox_links.file.isnull()), "retrievals remaining")
        vox_links.to_pickle(path1)
        return


class ArticleParser():
    """Parse article html to obtain text, title, author, and category"""
    def __init__(self):
        """Instantiate an article parser"""
        pass

    def parse(self, demo=True):
        """Perform the parse"""
        global vox
        if demo:
            if new_links_n != 0:
                df = vox_links.loc[:, ('link', 'date', 'file')]. \
                                    tail(new_links_n)
                vox = pd.concat([vox, df]).reset_index(drop=True)
                update = vox.tail(new_links_n).index.tolist()
            else:
                update = []
        else:
            vox = vox_links.loc[:, ('link', 'date', 'file')]   # copy vox_links
            update = vox[vox.file.isnull() == False].index.tolist()
        # parse
        for file in update:
            with open('files/'+str(file)+'.txt') as f:
                story_read = str(f.read())

            story_read = self._story_clean(story_read)   # clean bad encoding

            vox.loc[file, 'title'] = self._story_title(story_read)
            vox.loc[file, 'author'] = self._story_author(story_read)
            vox.loc[file, 'category'] = self._story_category(story_read)
            vox.loc[file, 'text'] = self._story_text(story_read)

        # Save the dataframe
        if demo:
            pass
        else:
            print('parsing complete')
        vox.to_pickle(path2)
        return

    def _story_clean(self, story_read):
        """Clean up bad encoding"""
        storyfix_pattern = r"""\\\w{3}"""
        story_read = re.sub(storyfix_pattern, '', story_read)
        story_read = story_read.replace('\\', '').replace('  ', ' ')
        return story_read

    def _story_title(self, story_read):
        """Find the article's title"""
        title_pattern = r"""<h1 class="c-page-title">([^<]*)</h1>"""
        story_title = re.findall(title_pattern, story_read)
        story_title = None if story_title == [] else story_title[0]
        return story_title

    def _story_author(self, story_read):
        """Find first author mentioned in the article"""
        author_pattern = r"""<a href="http://www.vox.com/authors/\S+>(\S+\s\S+\s?\S*)</a>"""
        story_author = re.findall(author_pattern, story_read)
        story_author = None if story_author == [] else story_author[0]
        return story_author

    def _story_category(self, story_read):
        """Find category of article"""
        category_pattern = r"""class="c-related-list__group-name">([a-zA-z\s\&]+)<"""
        story_category = re.findall(category_pattern, story_read)
        story_category = None if story_category == [] \
            else story_category[0].lower()
        return story_category

    def _story_text(self, story_read):
        """Find the text of the article"""
        story_content = story_read + ''   # make copy
        story_text = ''
        while (story_content.find('<p id=') != -1) \
                or (story_content.find('<p>') != -1):
            f1 = story_content.find('<p id=')
            f2 = story_content.find('<p>')
            if (f1 != -1) and (f2 != -1):
                m = min(f1, f2)
            elif (f1 != -1):
                m = f1
            else:
                m = f2

            story_content = story_content[m+1:]
            n = story_content.find('>')
            story_content = story_content[n+1:]
            o = story_content.find('</p>')
            paragraph = story_content[:o]
            story_content = story_content[o:]   # place to begin next loop.

            # fix paragraph
            paragraph2 = ''
            while paragraph.find('<') != -1:
                m = paragraph.find('<')
                paragraph2 += ' ' + paragraph[:m]   # add up to <
                paragraph = paragraph[m+1:]
                n = paragraph.find('>')   # if found, cutoff everything
                if n != -1:
                    paragraph = paragraph[n+1:]

            paragraph2 += ' ' + paragraph   # add anything remaining after cuts

            if (paragraph2.find('rn rn') == -1) \
                    and (paragraph2.find('embed-container') == -1):
                paragraph2 = paragraph2.replace('&nbsp;', ' ')\
                    .replace('&amp;', ' and ').replace('&#39;', "'")   # html
                paragraph2 = paragraph2.replace('&quot;', '"')
                paragraph2 = paragraph2.replace('&lt;', ' less than ')\
                    .replace('&gt;', ' greater than ')
                story_text += ' ' + paragraph2   # include blank between pars

        return None if story_text == '' else story_text


class Reviewer():
    """Tools to review Vox data"""
    def __init__(self):
        self.tokens = None
        self.token_counts = None
        self.token_set = None

    def drop_records(self, demo=True):
        """Drop Nulls.  Can only be run once after parser"""
        global vox
        if demo:
            vox = vox.dropna().reset_index(drop=True)
            sentences = vox.title.apply(lambda x: x[0:13] == 'Vox Sentences')
            vox = vox[sentences == False].reset_index(drop=True)
            if new_links_n != 0:
                for index in vox.tail(new_links_n).index.tolist():
                    vox.loc[index, 'date'] = self\
                        ._date_convert(vox.loc[index, 'date'])

        else:
            print('{} total articles'.format(len(vox)))
            print('---')
            self._missing()
            [self._missing(i) for i in ['title', 'author', 'category', 'text']]
            print('---')
            vox = vox.dropna().reset_index(drop=True)
            print('drop records with missing data')
            print('---')
            sentences = vox.title.apply(lambda x: x[0:13] == 'Vox Sentences')
            print('{} articles from "Vox Sentences" (daily summaries not articles)'
                  .format(sum(sentences)))
            vox = vox[sentences == False].reset_index(drop=True)
            print('drop "Vox Sentences"')
            print('---')
            vox.date = vox.date.apply(self._date_convert)
            print('date converted to datetime object')
            vox['year_month'] = vox.date.apply(self._yr_mo)
            print('year-month column added to dataframe')
            print('---')
            print('{} articles saved to new dataframe'.format(len(vox)))

        # Save the dataframe
        vox.to_pickle(path2)

        return

    def _missing(self, column=None):
        """Print the number of records with missing data"""
        if column is None:
            print('{} articles with missing data'
                  .format(len(vox) - len(vox.dropna())))
        else:
            print('{} missing {}'.format(sum(vox[column].isnull()), column))
        return

    def _date_convert(self, text):
        """Convert text to datetime.date object"""
        a, b, c = text.split('/')
        a, b, c = int(a), int(b), int(c)
        return datetime.date(a, b, c)

    def _yr_mo(self, date):
        """Obtain yr-mo from datetime"""
        joiner = '-0' if date.month <= 9 else '-'
        return str(date.year) + joiner + str(date.month)

    def search(self, pattern, column=None):
        """Search the text of a dataframe column for a pattern"""
        if column is None:
            return
        results = vox[vox[column].apply(lambda x:self
                                        ._find_pattern(pattern, x))]
        print('{} results found'.format(len(results)))
        return results

    def _find_pattern(self, pattern, text):
        """Return True if the pattern is in the text"""
        text = str(text)
        return False if re.search(pattern, text) is None else True

    def print_record(self, record):
        """Print link, title, author, category, date, text"""
        for item in ['link', 'title', 'author', 'category', 'date', 'text']:
                print(vox[item].loc[record])
        return
        pass

    def freq_plot(self, column=None, n=None, sort=False):
        """Create a frequency plot for the given column"""
        if column is None:
            return
        if n is None:
            if sort:
                pd.DataFrame(vox[column].value_counts()).sort_index()\
                    .plot.bar(figsize=(9, 4), legend=False)
            else:
                pd.DataFrame(vox[column].value_counts())\
                    .plot.bar(figsize=(9, 4), legend=False)
            plt.title('Number of Articles per {}'.format(column.title()))
        else:
            if sort:
                pd.DataFrame(vox[column].value_counts()).sort_index()[0:n]\
                    .plot.bar(figsize=(9, 4), legend=False)
            else:
                pd.DataFrame(vox[column].value_counts())[0:n]\
                    .plot.bar(figsize=(9, 4), legend=False)
            plt.title('Number of Articles per {} (n={})'
                      .format(column.title(), n))

        plt.xlabel(column.title())
        plt.ylabel('Number of Articles')
        return

    def tokenize(self):
        """Tokennize, Stem, Lemmatize"""
        pattern = '(?u)\\b\\w\\w+\\b'   # from sklearn.feature_extraction

        def tokenizer(x):
            """Function to tokenize text"""
            return ' '.join(re.findall(pattern, x)).lower()

        snowball = SnowballStemmer('english')

        def stemmer(x):
            """Function to stem tokens"""
            return ' '.join([snowball.stem(i) for i in x.split()])

        wordnet = WordNetLemmatizer()

        def lemmatizer(x):
            """Function to lemmatize tokens"""
            return ' '.join([wordnet.lemmatize(i) for i in x.split()])

        vox['tokened'] = vox.text.apply(tokenizer)
        vox['stemmed'] = vox.tokened.apply(stemmer)
        vox['lemmatized'] = vox.tokened.apply(lemmatizer)
        # Print number of tokens
        for column in ['tokened', 'stemmed', 'lemmatized']:
            self._tokens(column)
            print('{} tokens in "{}"'.format(len(
                self.token_counts.keys()), column))
        # Save the dataframe
        vox.to_pickle(path2)
        return

    def _tokens(self, column):
        """Count the tokens in a column"""
        if column is None:
            return
        tokens = vox[column].tolist()
        tokens = [item for row in tokens for item in row.split()]  # flatten
        token_counts = Counter(tokens)
        self.tokens = tokens
        self.token_counts = token_counts
        return

    def token_histogram(self):
        """Plot a histogram of the number of tokens per article"""
        token_count = vox.tokened.apply(lambda x: len(x.split()))
        token_count.hist(bins=100)
        plt.title('Histogram of Tokens per Article')
        plt.ylabel('Number of Tokens')
        return

    def zipfs(self):
        """Plot zipf's law based on our corpus"""
        self._tokens('tokened')
        print('{} total tokens'.format(len(self.tokens)))
        print('{} unique tokens'.format(len(self.token_counts.keys())))
        print('Top 5:', self.token_counts.most_common(5))

        y = sorted(list(self.token_counts.values()), reverse=True)
        i = np.arange(1, len(y)+1)
        plt.scatter(i, y)
        plt.title("Zipf's Law")
        plt.xlabel('Rank of Unique Token')
        plt.ylabel('Token Count')
        plt.show()
        plt.loglog(i, y)
        plt.title("Log/Log Zipf's Law")
        plt.xlabel('Log: Rank of Unique Token')
        plt.ylabel('Log: Token Count')
        plt.show()
        return

    def heaps(self):
        """Plot heaps' law based on our corpus (must run zipfs first)"""
        token_set = set()
        heaps_data = []
        uniq_tokens = 0
        for i, token in enumerate(self.tokens, 1):
            if token not in token_set:
                token_set.add(token)
                uniq_tokens += 1
            heaps_data.append((i, uniq_tokens))
        self.token_set = token_set

        heaps_data2 = []   # shorten list so that plotting is faster
        for i, item in enumerate(heaps_data):
            if i % 10000 == 0:
                heaps_data2.append(item)

        x, y = zip(*heaps_data2)
        plt.scatter(x, y)
        plt.title("Heaps' Law")
        plt.xlabel('Number of Tokens')
        plt.ylabel('Number of Unique Tokens')
        plt.xticks(rotation=45)
        plt.xlim((0, 7000000))
        return

    def compare_unigrams(self, n=5):
        """Compare our vocabulary to "unigram word counts"
        (must run zipfs and heaps first)"""
        unigram_link = "/Users/michaelseeber/Documents/Projects/DSCI6004-student/corpora/unigram_word_counts.txt"
        unigram = pd.read_csv(unigram_link, sep='\t', names=['word', 'count'])
        unigram_set = set(unigram.word.values.tolist())

        print('{} unique unigrams'.format(len(unigram_set)))
        print('{} unique tokens in Vox'.format(len(self.token_set)))

        vox_only = list(self.token_set.difference(unigram_set))
        print('{} tokens in Vox and not unigrams'.format(len(vox_only)))

        # eliminate tokens with digits
        vox_only = [x for x in vox_only if re.findall(r'\d', x) == []]
        vox_only_counter = Counter({token: self.token_counts[token]
                                    for token in vox_only})
        print('Top 5 unique to Vox:', vox_only_counter.most_common(5))
        return


def category_samples(n=15):
    """Print random sample of titles from each of the categories"""
    categories = vox.category.value_counts().index   # (sorted)
    for item in categories:
        print(item.upper())
        titles = vox.title[vox.category == item]
        if len(titles) > n:
            titles = np.random.choice(titles, size=n, replace=False)
        for title in titles:
                print(title)
        print('\n')
    return


class Splitter():
    """Set Global Train/Test split"""
    def __init__(self):
        """Save the train/test indices to be used with the split function"""
        self.alone = ['policy & politics', 'world', 'identities']
        self.together = ['culture', 'science & health', 'energy & environment',
                         'technology', 'new money', 'business & finance']
        self.keep_df = vox[vox.category.isin(self.alone + self.together)]
        indices = self.keep_df.index.values
        size = int(round(len(indices) * .65, 0))
        self.train = set(np.random.choice(indices, size, replace=False))
        self.test = set(indices).difference(self.train)

    def split(self, Xcolumn='tokened', pr=True):
        """Perform the split with the identified Xcolumns"""
        global X_train, X_test, y_train, y_test
        X_train = self.keep_df[Xcolumn].loc[self.train].values
        X_test = self.keep_df[Xcolumn].loc[self.test].values
        y_train = self.keep_df.category.isin(self.together).loc[self.train]\
            .values * 1
        y_test = self.keep_df.category.isin(self.together).loc[self.test]\
            .values * 1
        if pr:
            print('Split(size): X_train({}), X_test({}), y_train({}), ytest({})'
                  .format(len(X_train), len(X_test), len(y_train), len(y_test)))
        return

    def sample(self, n, pr=True):
        """Reduce training/test splits to a small sample."""
        global X_train, y_train
        random_train = np.random.choice(range(len(X_train)), size=n,
                                        replace=False)
        X_train = X_train[random_train]
        y_train = y_train[random_train]
        if pr:
            print('Sample(size): X_train({}), y_train({})'
                  .format(len(X_train), len(y_train)))
        return


class SparseArray():
    """Fit/Transform sparse array to normal array"""
    def __init__(self, param=True):
        """arbitarry param needed for gridsearch"""
        self.param = param

    def fit(self, X, y):
        """Return self so pipeline can call SparseArray().fit().transform()"""
        return self

    def transform(self, X):
        """Perform the transformation"""
        if type(X) is not np.ndarray:
            return X.toarray()
        else:
            return X

    def get_params(self, deep=True):
        """Required for gridsearch"""
        return {'param': self.param}


class ModelPipe():
    """End-to-end pipeline for a model"""
    def __init__(self, name,
                 vectorizer=TfidfVectorizer(ngram_range=(1, 2),
                                            stop_words='english',
                                            max_df=0.95,
                                            max_features=None,
                                            min_df=2, binary=False,
                                            use_idf=True,
                                            sublinear_tf=False),
                 vectorizer_grid={},
                 standard_scalar=False,
                 decomposition=False,
                 decomposition_grid={},
                 model=None,
                 model_grid={},
                 search='grid',
                 search_n=None,
                 search_validation=StratifiedShuffleSplit(n_splits=1,
                                                          test_size=0.35,
                                                          random_state=32),
                 search_score='accuracy'):
        """Define the pipeline"""
        self.name = name
        # Feature Extraction
        self.vectorizer = vectorizer
        self.vectorizer_grid = vectorizer_grid
        # Feature Transformation
        self.standard_scalar = standard_scalar
        self.decomposition = decomposition
        self.decomposition_grid = decomposition_grid
        # Model
        self.model = model
        self.model_grid = model_grid
        # Search
        self.search = search
        self.search_n = search_n   # n for randomized search
        self.search_validation = search_validation
        self.search_score = search_score


class ModelSearch():
    """Search a list of models and output all results"""
    def __init__(self, model_list, pr=True):
        """Provide a list of models"""
        self.model_list = model_list
        self.model_results_ = {}
        self.pr = pr

    def search(self):
        """Search all the models"""
        results = []
        self.best_models_ = []
        for model in self.model_list:
            if self.pr:
                print('\n')
                print(model.name)
            result = self._fit(model, X_train, y_train)
            result = self._result(model, result)
            results.append(result)

        self.model_results_ = self._results(results)
        if self.pr:
            print('Done!')

    def _fit(self, model, X, y):
        """Perform the grid or random search"""
        pipe, grid = self._pipeline(model)
        searcher = self._searcher(model, pipe, grid)
        searcher.fit(X, y)
        self.best_models_.append(searcher)
        return searcher.cv_results_

    def _pipeline(self, model):
        """Create the pipeline"""
        pipe = [('model', model.model)]
        grid = {'model__'+k: v for k, v in model.model_grid.items()}

        if model.vectorizer is not None:
            pipe.insert(-1, ('vectorizer', model.vectorizer))
            grid.update({'vectorizer__'+k: v for k, v in
                         model.vectorizer_grid.items()})

        if (model.standard_scalar is not False) \
            or (model.decomposition.__class__ ==
                sklearn.decomposition.pca.PCA) \
            or (model.model.__class__ ==
                sklearn.discriminant_analysis.LinearDiscriminantAnalysis) \
            or (model.model.__class__ ==
                sklearn.ensemble.gradient_boosting.GradientBoostingClassifier):
            pipe.insert(-1, ('sparse', SparseArray()))

        if model.standard_scalar is not False:
            pipe.insert(-1, ('scalar', StandardScaler()))

        if model.decomposition is not False:
            pipe.insert(-1, ('decomposition', model.decomposition))
            grid.update({'decomposition__'+k: v for k, v in
                         model.decomposition_grid.items()})

        pipe = Pipeline(pipe)
        return pipe, grid

    def _searcher(self, model, pipe, grid):
        """Select grid or randomized search"""
        # Number of permutations in grid
        grid_perm = np.product(np.asarray([len(v) for v in grid.values()]))

        if (model.search == 'grid') or (model.search_n >= grid_perm):
            return GridSearchCV(pipe, grid,
                                cv=model.search_validation,
                                scoring=model.search_score,
                                n_jobs=4, verbose=self.pr)
        else:
            return RandomizedSearchCV(pipe, grid, n_iter=model.search_n,
                                      cv=model.search_validation,
                                      scoring=model.search_score,
                                      n_jobs=4, verbose=self.pr)

    def _result(self, model, result):
        """Add model name to result set"""
        n = len(result['mean_fit_time'])
        result['_pipe'] = [model.name] * n
        return result

    def _results(self, results):
        """Append the results"""
        df = pd.DataFrame(results[0])
        for result in results[1:]:
            df = df.append(pd.DataFrame(result))
        return df.reset_index(drop=True)


def plot_vectorizer_results(results):
    """Plot results of the vectorizer search"""
    y = results['mean_test_score'].sort_values(ascending=False).values
    x = list(range(1, len(y)+1))
    plt.scatter(x, y)
    plt.xlabel('Different Vectorizers')
    plt.ylabel('Mean Test Score')
    plt.title('Mean Test Score by Vectorizer')
    return


def plot_Knn_results(results):
    """Plot the mean test score by number of neighbors"""
    plt.plot(results.param_model__n_neighbors.values,
             results.mean_test_score.values)
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Mean Test Score')
    plt.title('Mean Test Score by K')
    return


def results_table(results):
    """Create a results table from the results of a ModelSearch"""
    df = results.loc[:, ('_pipe', 'mean_fit_time', 'mean_score_time',
                         'mean_test_score')]
    df['Mean Test Score'] = df['mean_test_score']
    df['Mean Run Time'] = df['mean_fit_time'] + df['mean_score_time']
    df['Pipe'] = df['_pipe']
    return df.drop(['_pipe', 'mean_fit_time', 'mean_score_time',
                   'mean_test_score'], axis=1).set_index('Pipe') \
        .sort_values(by='Mean Test Score', ascending=False)


class Ensemble():
    """Fit ensemblers to your models"""
    def __init__(self, model_list, ensemble_list, samples):
        """Pass a list of models and a list of ensemblers"""
        self.model_list = model_list
        self.ensemble_list = ensemble_list
        self.samples = samples

    def fit(self):
        """Fit models and ensembles to each sample and collect results"""
        self._features()
        self.models_ensembles_ = [model[1] for model in self.model_list]
        self.names_ = [model[0] for model in self.model_list]
        ensembles_only = []
        for ensemble in self.ensemble_list:
            ensemble_name, ensemble_model = ensemble
            ensemble_model.fit(self.X_train_ensemble_, y_train)
            self.models_ensembles_.append(ensemble_model)
            ensembles_only.append(ensemble_model)
            self.names_.append(ensemble_name)

        self.results_ = []
        for i in range(self.samples):
            size = len(self.X_test_ensemble_)
            sample = np.random.choice(range(size), size=size, replace=True)
            Xi = X_test[sample]
            yi = y_test[sample]
            Xi_ensemble = []
            result = []
            for model in self.model_list:
                yi_pred = model[1].predict(Xi)
                Xi_ensemble.append(yi_pred)
                result.append(accuracy_score(yi, yi_pred))
            Xi_ensemble = np.asarray(Xi_ensemble).T
            for model in ensembles_only:
                yi_pred = model.predict(Xi_ensemble)
                result.append(accuracy_score(yi, yi_pred))
            self.results_.append(result)
        return

    def _features(self):
        """Create features for ensembles"""
        self.X_train_ensemble_ = [model[1].predict(X_train)
                                  for model in self.model_list]
        self.X_test_ensemble_ = [model[1].predict(X_test)
                                 for model in self.model_list]
        self.X_train_ensemble_ = np.asarray(self.X_train_ensemble_).T
        self.X_test_ensemble_ = np.asarray(self.X_test_ensemble_).T

    def model_correlation(self):
        """Test correlation of model predictions"""
        return pd.DataFrame(self.X_test_ensemble_, columns=[model[0]
                            for model in self.model_list]).corr()

    def results_table(self):
        """Summary statistics for test results from each model/sample"""
        return pd.DataFrame(self.results_, columns=self.names_)\
            .describe().T[['mean', 'std', 'min', 'max']]\
            .sort_values(by='mean', ascending=False)


class MakePredictions():
    """Make predictions and add them to vox dataframe"""
    def __init__(self, missing=True):
        """Specify predictions on all data or those missing preds."""
        self.missing = missing
        self.model = joblib.load(path3)

    def predict(self):
        """Perform the predictions"""
        global vox
        if self.missing:
            indices = vox[vox.vox_label.isnull()].index
        else:
            indices = vox.index

        for item in indices:
            vox.loc[item, 'vox_label'] = self._vox_label(
                                            vox.category.loc[item])
            vox.loc[item, 'predicted_label'] = self._predicted_label(
                                                vox.lemmatized.loc[item])

        # Save the dataframe
        vox.to_pickle(path2)
        return

    def _vox_label(self, category):
        """Obtain the Vox label for an article"""
        alone = ['policy & politics', 'world', 'identities']
        together = ['culture', 'science & health', 'energy & environment',
                    'technology', 'new money', 'business & finance']
        if category in alone:
            return "Enjoy Alone"
        elif category in together:
            return "Enjoy Together"
        else:
            return "Unknown"

    def _predicted_label(self, text):
        """Obtain the predicted label for an article"""
        text = [text]
        prediction = self.model.predict(text)
        return "Enjoy Alone" if prediction == 0 else "Enjoy Together"


def confusion_matrix(test_df):
    """Print a confusion matrix"""
    ct = pd.crosstab(test_df.vox_label, test_df.predicted_label)
    TP = ct['Enjoy Together']['Enjoy Together']
    FP = ct['Enjoy Together']['Enjoy Alone']
    TN = ct['Enjoy Alone']['Enjoy Alone']
    FN = ct['Enjoy Alone']['Enjoy Together']
    print('True Negatives', TN, '({})'
          .format(round(TN/(TN+FP+FN+TP), 3)), sep='\t')
    print('False Positives', FP, '({})'
          .format(round(FP/(TN+FP+FN+TP), 3)), '(Type I Error)', sep='\t')
    print('False Negatives', FN, '({})'
          .format(round(FN/(TN+FP+FN+TP), 3)), '(Type II Error)', sep='\t')
    print('True Positives', TP, '({})'
          .format(round(TP/(TN+FP+FN+TP), 3)), sep='\t')
    print('Accuracy', round((TN+TP)/(TN+FP+FN+TP), 3), sep='\t')
    print('Error',  round((FN+FP)/(TN+FP+FN+TP), 3), sep='\t\t')
    print('Recall',  round((TP)/(TP+FN), 3), sep='\t\t')
    print('Precision',  round((TP)/(TP+FP), 3), sep='\t')
    print('F1', round(2 / ((1/((TP)/(TP+FP))) + (1/((TP)/(TP+FN)))), 3),
          sep='\t\t')
    print('\nConfusion Matrix:')
    return ct


def clustering(n_clusters=(5, 5)):
    """Perform clustering on each class"""
    for j, label in enumerate(["Enjoy Alone", "Enjoy Together"]):
        X = vox.lemmatized[vox.predicted_label == label].values
        vectorizer = TfidfVectorizer(ngram_range=(1, 1),
                                     stop_words='english',
                                     max_df=0.95,
                                     max_features=None,
                                     min_df=2,
                                     binary=False,
                                     use_idf=True,
                                     sublinear_tf=False)
        nmf = NMF(init="nndsvd", n_components=n_clusters[j], max_iter=200)
        pipe = Pipeline([('vectorizer', vectorizer), ('nmf', nmf)])
        pipe.fit(X)

        W = pipe.transform(X)
        H = pipe.named_steps['nmf'].components_

        terms = [""] * len(pipe.named_steps['vectorizer'].vocabulary_)
        for term in pipe.named_steps['vectorizer'].vocabulary_.keys():
            terms[pipe.named_steps['vectorizer'].vocabulary_[term]] = term

        for i, row in enumerate(H, 1):
            r = row.tolist()
            x = list(zip(r, terms))
            s = sorted(x, reverse=True)
            value, term = zip(*s)
            print(label, i, term[0:10])
        print('\n')

        # Save to file
        path = path4 if j == 0 else path5
        joblib.dump(pipe, path)
    return


class AssignClusters():
    """Make clusters and add them to vox dataframe
    RUN AFTER MakePredictions"""
    def __init__(self, missing=True):
        """Make predictions on all vox or missing clusters."""
        self.missing = missing
        self.alone = joblib.load(path4)
        self.together = joblib.load(path5)

    def assign(self):
        """Assign each article to a cluster"""
        global vox
        if self.missing:
            indices = vox[vox.cluster.isnull()].index
        else:
            indices = vox.index

        for item in indices:
            label = vox.loc[item, 'predicted_label']
            vox.loc[item, 'cluster'] = self._cluster(label,
                                                     vox.lemmatized.loc[item])

        # Save the dataframe
        vox.to_pickle(path2)
        return

    def _cluster(self, label, text):
        """Determine the cluster for the article"""
        text = [text]
        if label == "Enjoy Alone":
            return "A" + str(np.argmax(self.alone.transform(text)[0]) + 1)
        else:
            return "T" + str(np.argmax(self.together.transform(text)[0]) + 1)


class SearchEngine():
    """Build store and query a search engine"""
    def __init__(self):
        """Create and query a search engine"""
        assert vox.tail(1).index + 1 == len(vox)
        self.vectorizer = joblib.load(path6)
        self.vectorized = joblib.load(path7)

    def build(self):
        """Build the vectorizer and vectorized data"""
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1),
                                          stop_words='english',
                                          max_df=1.0,
                                          max_features=None,
                                          min_df=1,
                                          binary=False,
                                          use_idf=True,
                                          sublinear_tf=False)
        self.vectorizer.fit(vox.lemmatized)
        self.vectorized = self.vectorizer.transform(vox.lemmatized)
        joblib.dump(self.vectorizer, path6)
        joblib.dump(self.vectorized, path7)
        return

    def check_tfidf(self, word):
        """Check the tfidf of a given word"""
        word_index = self.vectorizer.vocabulary_[word]   # index of a word
        print('{} is the idf for the word'
              .format(round(self.vectorizer.idf_[word_index], 3)))
        print('{} is the number of articles with the word'
              .format(len(self.vectorized[:, word_index].data)))
        article = np.argmax(self.vectorized[:, word_index].toarray())
        print('{} is the max tfidf for this word across all articles'
              .format(round(self.vectorized[article, word_index], 3)))
        print('"{}" is the article with the max tfidf '
              .format(vox.title.loc[article]))
        return

    def search(self, phrase):
        """Perform a search"""
        pattern = r'(.*)([\<\>\^])(.*)'
        if re.findall(pattern, phrase) == []:
            n = ''
        else:
            phrase, cluster, n = re.findall(pattern, phrase)[0]

        if phrase != '':   # perform phrase search, n is num articles
            phrase = self._lemmatize_phrase(phrase)
            vectorized_phrase = self.vectorizer.transform([phrase])
            cosine_sim = self.vectorized.dot(vectorized_phrase.T)

            df = vox[['title', 'author', 'link', 'date']]
            df = df.join(pd.Series(cosine_sim.toarray().reshape(-1),
                                   name='cosine'))
            df = df.sort_values(by='cosine', ascending=False)

            today = max(df.date)
            week = today - datetime.timedelta(days=6)
            df['time_frame'] = df.date.apply(lambda x:
                                             self._time_frame(x, today, week))

            n = 3 if n == '' else int(n)
            self._print_result(df[(df.time_frame == 'TODAY') &
                                  (df.cosine != 0)].head(n))
            self._print_result(df[(df.time_frame == 'THIS WEEK') &
                                  (df.cosine != 0)].head(n))
            self._print_result(df[(df.time_frame == 'ALL TIME') &
                                  (df.cosine != 0)].head(n))

        elif cluster == '^':   # Latest articles, n is the number of days.
            df = vox[['title', 'author', 'link', 'date']]
            n = 1 if n == '' else int(n)
            date_range = max(df.date) - datetime.timedelta(days=n)
            df = df[df.date > date_range]
            df = df.sort_values(by='date', ascending=False)
            self._print_result(df)

        elif (cluster == '<') or (cluster == '>'):   # 10 articles
            alone = ['A1', 'A2', 'A3', 'A4', 'A5',
                     'A6', 'A7', 'A8', 'A9', 'T5']   # reassign T5
            together = ['T1', 'T2', 'T3', 'T4', 'T6',
                        'T7', 'T8', 'T9', 'T10', 'T11']
            df = vox[['title', 'author', 'link', 'date', 'cluster']]
            if cluster == '<':
                df = df[df.cluster.isin(alone)]\
                    .sort_values(by='date', ascending=False)
            else:
                df = df[df.cluster.isin(together)]\
                    .sort_values(by='date', ascending=False)
            df = df.groupby('cluster').head(1)
            self._print_result(df)

        else:
            pass
        return

    def _lemmatize_phrase(self, phrase):
        """Lemmatize the queried phraase"""
        pattern = '(?u)\\b\\w\\w+\\b'

        def tokenizer(x):
            return ' '.join(re.findall(pattern, x)).lower()

        wordnet = WordNetLemmatizer()

        def lemmatizer(x):
            return ' '.join([wordnet.lemmatize(i) for i in x.split()])

        return lemmatizer(tokenizer(phrase))

    def _time_frame(self, date, today, week):
        """Obtain timeframe of each article"""
        if date == today:
            return 'TODAY'
        elif date >= week:
            return 'THIS WEEK'
        else:
            return 'ALL TIME'

    def _print_result(self, df):
        """Print the results"""
        if len(df) > 0:
            try:
                print('{}'.format(df.time_frame.iloc[0]))
            except Exception:
                pass
            for i in range(len(df)):
                print('{}) {}'.format(i+1, df.title.iloc[i]))
                print('   {} | {}'.format(df.author.iloc[i], df.date.iloc[i]))
                print('   {}'.format(df.link.iloc[i]))
            print('\n')
        return


def demo():
    """Under Construction"""
    link_scraper = LinkScraper(start_page=1, stop_page=15)
    link_scraper.scrape(demo=True)

    article_retriever = ArticleRetriever(number=100)
    article_retriever.retrieve(demo=True)

    article_parser = ArticleParser()
    article_parser.parse()
