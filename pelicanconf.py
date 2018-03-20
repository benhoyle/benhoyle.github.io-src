#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'Ben Hoyle'
SITENAME = u'Machine Learning Projects'
SITEURL = ''

THEME = 'simple'

PATH = 'content'

TIMEZONE = 'Europe/London'
DEFAULT_DATE = 'fs'

USE_FOLDER_AS_CATEGORY = True

DEFAULT_LANG = u'en'

# Set default source for slug metadata
SLUGIFY_SOURCE = 'basename'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Portfolio Details
PORTFOLIO_TITLE = "Project Portfolio"
PORTFOLIO_SUBTITLE = "A selection of machine learning projects"

# Blogroll
LINKS = (('Pelican', 'http://getpelican.com/'),
         ('Python.org', 'http://python.org/'),
         ('Jinja2', 'http://jinja.pocoo.org/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
