# Suppa
Project Suppa is a suite of programs that automates weekly meal planning. Our meal planning consists of four main steps:

1. recipe sourcing and development
2. splitting prep work
3. shopping list generation (SuppaList)
4. grocery shopping

Only step 3, SuppaList, is currently implemented.

## SuppaList
SuppaList generates a shopping list from recipe ingedient lists. It combines the recipe lists into a single list, sorts ingredients by aisle, and combines like ingredients.

SuppaList runs entirely in the browser. You can use it at https://www.voegele.me/posts/suppalist/

![suppalist](https://github.com/user-attachments/assets/7ce575bc-87be-4cb5-b07a-ce7f857c148e)

# Building
## suplistml
```
./build.py
```

## suplistmslrs
```
just
```

## suplistjs
```
npm run all
```

# Training
```
$ cd suplistml
$ python -m suplistml.model.train
```

# Copyright
Copyright (C) 2025 Chad Voegele. All Rights Reserved.

# License
This code is distributed under the terms of the "GNU GPLv2 only". See LICENSE file for details.

# Author
Chad Voegele
