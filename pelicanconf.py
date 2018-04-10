#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'Ben Hoyle'
SITENAME = u'Practical Machine Learning Adventures'
SITESUBTITLE = "A selection of machine learning projects"
SITEURL = ''

THEME = 'alchemy'

PATH = 'content'

TIMEZONE = 'Europe/London'
DEFAULT_DATE = 'fs'

USE_FOLDER_AS_CATEGORY = True

DEFAULT_LANG = u'en'

STATIC_PATHS = ['images', 'notebooks']

# Set default source for slug metadata
SLUGIFY_SOURCE = 'basename'

LINKS = ""
DISPLAY_PAGES_ON_MENU = False

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True

ARTICLE_ORDER_BY = 'title'

TWITTER_URL = "https://twitter.com/bjh_ip"
LINKEDIN_URL = "https://www.linkedin.com/in/benhoyle/"
