-- Every club seen since 2014-15, with each source's identifier.
select * from {{ ref('teams') }}
