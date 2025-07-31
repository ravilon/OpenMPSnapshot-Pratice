/*
 * Copyright (C) 2019  Ortega Froysa, Nicolás <nicolas@ortegas.org>
 * Author: Ortega Froysa, Nicolás <nicolas@ortegas.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <gmp.h>

struct llist_item {
	mpz_t n; // the number stored
	struct llist_item *next; // the next element in the list
	struct llist_item *prev; // the previous element in the list
};

struct llist {
	struct llist_item *first; // first element of the list
	struct llist_item *last; // last element of the list
	size_t size; // number of elements in the list
};

/*
 * initialize an empty linked list where `first` and `last` are
 * equal to NULL and `size` is 0.
 */
void llist_init(struct llist *list);

/*
 * free all space allocated to the list.
 */
void llist_deinit(struct llist *list);

/*
 * insert an item at the end of the linked list with value `n`.
 *
 * returns 1 on success, 0 on failure.
 */
int llist_insert(struct llist *list, mpz_t n);
/*
 * insert an item in the linked list in a sorted position.
 *
 * returns 1 on success, 0 on failure.
 */
int llist_sorted_insert(struct llist *list, mpz_t n);
