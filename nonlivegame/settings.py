from os import environ

SESSION_CONFIGS = [
    dict(
        name = 'chat_noavg',
        app_sequence = ['quiz','chat_noavg','chat_noavg2','chat_noavg3','chat_noavg4','survey','final'],
        num_demo_participants = 16,
        room = 'chat_noavg'
    ),
]

# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=0.25, participation_fee=7.50, doc=""
)

PARTICIPANT_FIELDS = []
SESSION_FIELDS = []

# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
LANGUAGE_CODE = 'en'

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True
ROOMS = [dict(name ='chat_noavg', display_name ='chat_noavg', participant_label_file='participant_labels.txt')]
ADMIN_USERNAME = 'admin'
# for security, best to set admin password in an environment variable
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')

DEMO_PAGE_INTRO_HTML = """ """

SECRET_KEY = '9707695087914'
