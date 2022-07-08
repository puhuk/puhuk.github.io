---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

Sangho Lee, Seoyoung Lee, Joonseok Lee. Towards Detailed Characteristic-Preserving Virtual Try-On, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), The 5th Workshop on Computer Vision for Fashion, Art, and Design, 2022.

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}
