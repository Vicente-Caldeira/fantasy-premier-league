{% test value_between(model, column_name, low, high) %}
select * from {{ model }} where {{ column_name }} not between {{ low }} and {{ high }}
{% endtest %}
