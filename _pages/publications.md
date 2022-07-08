---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

Towards Detailed Characteristic-Preserving Virtual Try-On (https://openaccess.thecvf.com/content/CVPR2022W/CVFAD/papers/Lee_Towards_Detailed_Characteristic-Preserving_Virtual_Try-On_CVPRW_2022_paper.pdf)

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}
