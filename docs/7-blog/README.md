---
layout: default
title: Blog
nav_order: 8
permalink: /7-blog/
---

# Blog

<div class="posts">
  {% for post in site.posts %}
    <article class="post">

      <h2><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h2>

      <div class="date">
        By <a href="{{ post.author_url }}">{{ post.author }}</a> - {{ post.date | date: "%B %e, %Y" }}
      </div>

      <div class="entry">
        {{ post.excerpt }}
      </div>

      <a href="{{ site.baseurl }}{{ post.url }}">Read More...</a>
    </article>
  {% endfor %}
</div>

